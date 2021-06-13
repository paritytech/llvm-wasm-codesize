//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the general function merging optimization.
//  
// It identifies similarities between functions, and If profitable, merges them
// into a single function, replacing the original ones. Functions do not need
// to be identical to be merged. In fact, there is very little restriction to
// merge two function, however, the produced merged function can be larger than
// the two original functions together. For that reason, it uses the
// TargetTransformInfo analysis to estimate the code-size costs of instructions
// in order to estimate the profitability of merging two functions.
//
// This function merging transformation has three major parts:
// 1. The input functions are linearized, representing their CFGs as sequences
//    of labels and instructions.
// 2. We apply a sequence alignment algorithm, namely, the Needleman-Wunsch
//    algorithm, to identify similar code between the two linearized functions.
// 3. We use the aligned sequences to perform code generate, producing the new
//    merged function, using an extra parameter to represent the function
//    identifier.
//
// This pass integrates the function merging transformation with an exploration
// framework. For every function, the other functions are ranked based their
// degree of similarity, which is computed from the functions' fingerprints.
// Only the top candidates are analyzed in a greedy manner and if one of them
// produces a profitable result, the merged function is taken.
// 
//===----------------------------------------------------------------------===//
//
// This optimization was proposed in
//
// Function Merging by Sequence Alignment (CGO'19)
// R. Rocha, P. Petoumenos, Z. Wang, M. Cole, H. Leather
//
// Effective Function Merging in the SSA Form (PLDI'20)
// R. Rocha, P. Petoumenos, Z. Wang, M. Cole, H. Leather
//
// HyFM: Function Merging for Free (LCTES'21)
// R. Rocha, P. Petoumenos, Z. Wang, M. Cole, K. Hazelwood, H. Leather
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FunctionMerging.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include <algorithm>
#include <list>
#include <set>
#include <unordered_set>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "func-merging"

STATISTIC(NumMergedFunctions, "Number of functions merged");


static cl::opt<unsigned> RankingThreshold(
    "func-merging-ranking-threshold", cl::init(0), cl::Hidden,
    cl::desc("Threshold of how many candidates should be ranked"));

static cl::opt<int> MergingOverheadThreshold(
    "func-merging-threshold", cl::init(0), cl::Hidden,
    cl::desc("Threshold of allowed overhead for merging function"));

static cl::opt<bool>
    ForceMerge("func-merging-force", cl::init(false), cl::Hidden,
                  cl::desc("Force all merges regardless of profitability"));

static cl::opt<bool>
    MaxParamScore("func-merging-max-param", cl::init(true), cl::Hidden,
                  cl::desc("Maximizing the score for merging parameters"));

static cl::opt<bool>
    EnableUnifiedReturnType("func-merging-unify-return", cl::init(false), cl::Hidden,
                  cl::desc("Enable unified return types"));

static cl::opt<bool>
    EnableOperandReordering("func-merging-operand-reorder", cl::init(true), cl::Hidden,
                  cl::desc("Enable operand reordering"));

static cl::opt<bool> ReuseMergedFunctions (
    "func-merging-reuse-merges", cl::init(true), cl::Hidden,
    cl::desc("Try to reuse merged functions for another merge operation"));

static cl::opt<unsigned> MaxNumSelection (
    "func-merging-max-selects", cl::init(500), cl::Hidden,
    cl::desc("Maximum number of allowed operand selection"));

struct MatchingBlocks {
  BasicBlock *Blocks[2];
  std::map<Instruction*, Instruction*> MatchingInsts;

  MatchingBlocks() {
    Blocks[0] = Blocks[1] = nullptr;
  }

  MatchingBlocks(BasicBlock *BB0, BasicBlock *BB1) {
    Blocks[0] = BB0;
    Blocks[1] = BB1;
  }

  BasicBlock *operator[](size_t i) { return Blocks[i]; }

  void setMatchingPair(Instruction *I1, Instruction *I2) {
    MatchingInsts[I1] = I2;
    MatchingInsts[I2] = I1;
  }

  bool isMatchingPair(Instruction *I1, Instruction *I2) {
    if (MatchingInsts.find(I1)==MatchingInsts.end()) return false;
    return (MatchingInsts[I1]==I2);
  }
  
  Instruction *getMatchingInstruction(Instruction *I) {
    if (MatchingInsts.find(I)==MatchingInsts.end()) return nullptr;
    return MatchingInsts[I];
  }
};

class CodeAlignment {
public:
  std::vector<MatchingBlocks> AlignedBlocks;
  CodeAlignment() {}
};

class PairwiseAlignment : public CodeAlignment {
public:
  PairwiseAlignment(ArrayRef<BasicBlock*> Blocks1, ArrayRef<BasicBlock*> Blocks2, const FunctionMergingOptions &Options = {});
};

class MergedInstruction {
public:
  Instruction *NewI;
  Instruction *Insts[2];

  MergedInstruction() : NewI(nullptr) {
    Insts[0] = Insts[1] = nullptr;
  }

  MergedInstruction(Instruction *NewI, Instruction *I1, Instruction *I2) : NewI(NewI) {
    Insts[0] = I1;
    Insts[1] = I2;
  }

  Instruction *get() { return NewI; }
  Instruction *operator[](size_t i) { return Insts[i]; }
};

class CodeMerger {
private:
  Module *M;
 
  const DataLayout *DL;
  LLVMContext *ContextPtr;
  Type *IntPtrTy;

  Value *IsFirst;

  ArrayRef<BasicBlock*> Blocks1;
  ArrayRef<BasicBlock*> Blocks2;

  BasicBlock *EntryBB1;
  BasicBlock *EntryBB2;
  BasicBlock *PreBB;

  Type *RetType1;
  Type *RetType2;
  Type *ReturnType;

  bool RequiresUnifiedReturn;

  Function *F;
  std::vector<SelectInst*> AllSelections;

  SmallPtrSet<BasicBlock*,8> CreatedBBs;
  SmallPtrSet<Instruction*,8> CreatedInsts;

  bool defineReturnType(Function *F1, Function *F2, const FunctionMergingOptions &Options = {});

  Value *mergeValues(Value *V1, Value *V2, Instruction *InsertPt);
  bool assignOperands(Instruction *I, bool IsFuncId1, ValueToValueMapTy &VMap);
  bool assignLabelOperands(Instruction *I, std::unordered_map<BasicBlock*, BasicBlock *> &BlocksReMap, ValueToValueMapTy &VMap);
  bool assignPHIOperandsInBlock(BasicBlock *BB, std::unordered_map<BasicBlock*, BasicBlock *> &BlocksReMap, ValueToValueMapTy &VMap);
  void storeInstIntoAddr(Instruction *IV, Value *Addr);
  AllocaInst* memfyInst(Instruction *I);


  Instruction *cloneInst(IRBuilder<> &Builder, Function *MF, Instruction *I);

  void codeGen(CodeAlignment *AR,
                    ValueToValueMapTy &VMap,
                    std::unordered_map<BasicBlock *, BasicBlock *> &BlocksF1,
                    std::unordered_map<BasicBlock *, BasicBlock *> &BlocksF2,
                    std::list<MergedInstruction> &MergedInsts
		    );
public:
  CodeMerger(Module *M, ArrayRef<BasicBlock*> Blocks1, ArrayRef<BasicBlock*> Blocks2) : M(M), Blocks1(Blocks1), Blocks2(Blocks2) {
    if (M) {
      DL = &M->getDataLayout();
      ContextPtr = &M->getContext();
      IntPtrTy = DL->getIntPtrType(*ContextPtr);
    }
  }

  FunctionMergeResult defineMergedFunction(Function *F1, Function *F2, const char *Name, CodeAlignment *AR, ValueToValueMapTy &VMap, const FunctionMergingOptions &Options = {});

  CodeMerger &setCondition(Value *IsFirst) {
    this->IsFirst = IsFirst;
    return *this;
  }

  CodeMerger &setEntryPoints(BasicBlock *EntryBB1, BasicBlock *EntryBB2) {
    this->EntryBB1 = EntryBB1;
    this->EntryBB2 = EntryBB2;
    return *this;
  }

  CodeMerger &setReturnTypes(Type *RetType1, Type *RetType2) {
    this->RetType1 = RetType1;
    this->RetType2 = RetType2;
    return *this;
  }

  CodeMerger &setEntryPoint(BasicBlock *PreBB) {
    this->PreBB = PreBB;
    return *this;
  }

  CodeMerger &setReturnType(Type *ReturnType, bool RequiresUnifiedReturn=false) {
    this->ReturnType = ReturnType;
    this->RequiresUnifiedReturn = RequiresUnifiedReturn;
    return *this;
  }

  CodeMerger &setFunction(Function *F) {
    this->F = F;
    return *this;
  }

  Function *getFunction() { return F; }
  Type *getReturnType() { return ReturnType; }
  bool getRequiresUnifiedReturn() { return RequiresUnifiedReturn; }

  Value *getCondition() { return IsFirst; }

  LLVMContext &getContext() { return *ContextPtr; }

  Type *getIntPtrType() { return IntPtrTy; }

  ArrayRef<BasicBlock*> getBlocks1() { return Blocks1; }
  ArrayRef<BasicBlock*> getBlocks2() { return Blocks2; }

  BasicBlock *getEntryBlock1() { return EntryBB1; }
  BasicBlock *getEntryBlock2() { return EntryBB2; }
  BasicBlock *getPreBlock() { return PreBB; }

  Type *getReturnType1() { return RetType1; }
  Type *getReturnType2() { return RetType2; }

  void insert(BasicBlock *BB) { CreatedBBs.insert(BB); }
  void insert(Instruction *I) { CreatedInsts.insert(I); }

  void erase(BasicBlock *BB) { CreatedBBs.erase(BB); }
  void erase(Instruction *I) { CreatedInsts.erase(I); }

  bool generate(CodeAlignment *AR,
                ValueToValueMapTy &VMap,
                const FunctionMergingOptions &Options = {});

  void destroyGeneratedCode();

  SmallPtrSet<Instruction*,8>::const_iterator begin() const { return CreatedInsts.begin(); }
  SmallPtrSet<Instruction*,8>::const_iterator end() const { return CreatedInsts.end(); }

};

bool matchBlockEntry(BasicBlock *BB1, BasicBlock *BB2);
bool matchInstructions(Instruction *I1, Instruction *I2, const FunctionMergingOptions &Options = {});
bool validMergeTypes(Function *F1, Function *F2, const FunctionMergingOptions &Options = {});

class FunctionMerging {
public:
  bool runImpl(Module &M);
};

static bool matchLandingPad(LandingPadInst *LP1, LandingPadInst *LP2) {
  if (LP1->getType() != LP2->getType())
    return false;
  if (LP1->isCleanup() != LP2->isCleanup())
    return false;
  if (LP1->getNumClauses() != LP2->getNumClauses())
    return false;
  for (unsigned i = 0; i < LP1->getNumClauses(); i++) {
    if (LP1->isCatch(i) != LP2->isCatch(i))
      return false;
    if (LP1->isFilter(i) != LP2->isFilter(i))
      return false;
    if (LP1->getClause(i) != LP2->getClause(i))
      return false;
  }
  return true;
}

static bool matchGetElementPtrInsts(const GetElementPtrInst *GEP1, const GetElementPtrInst *GEP2) {
  Type *Ty1 = GEP1->getSourceElementType();
  SmallVector<Value*, 16> Idxs1(GEP1->idx_begin(), GEP1->idx_end());

  Type *Ty2 = GEP2->getSourceElementType();
  SmallVector<Value*, 16> Idxs2(GEP2->idx_begin(), GEP2->idx_end());

  if (Ty1!=Ty2) return false;
  if (Idxs1.size()!=Idxs2.size()) return false;

  if (Idxs1.empty())
    return true;

  for (unsigned i = 1; i<Idxs1.size(); i++) {
    Value *V1 = Idxs1[i];
    Value *V2 = Idxs2[i];

    //structs must have constant indices, therefore they must be constants and must be identical when merging
    if (isa<StructType>(Ty1)) {
      if (V1!=V2) return false;
    }
    Ty1 = GetElementPtrInst::getTypeAtIndex(Ty1, V1);
    Ty2 = GetElementPtrInst::getTypeAtIndex(Ty2, V2);
    if (Ty1!=Ty2) return false;
  }
  return true;
}

static bool matchSwitchInsts(const SwitchInst *SI1, const SwitchInst *SI2) {
  if (SI1->getNumCases() == SI2->getNumCases()) {
    auto CaseIt1 = SI1->case_begin(), CaseEnd1 = SI1->case_end();
    auto CaseIt2 = SI2->case_begin(), CaseEnd2 = SI2->case_end();
    do {
      auto *Case1 = &*CaseIt1;
      auto *Case2 = &*CaseIt2;
      if (Case1 != Case2)
        return false; // TODO: could allow permutation!
      ++CaseIt1;
      ++CaseIt2;
    } while (CaseIt1 != CaseEnd1 && CaseIt2 != CaseEnd2);
    return true;
  }
  return false;
}

static bool matchInsertValueInsts(const InsertValueInst *IV1, const InsertValueInst *IV2) {
    return IV1->getIndices() == IV2->getIndices();
}

static bool matchExtractValueInsts(const ExtractValueInst *EV1, const ExtractValueInst *EV2) {
    return EV1->getIndices() == EV2->getIndices();
}

static bool matchCallInsts(const CallBase *CI1, const CallBase *CI2) {
  if (CI1->isInlineAsm() || CI2->isInlineAsm())
    return false;

  if (CI1->getCalledFunction() != CI2->getCalledFunction())
    return false;
 
  if (Function *F = CI1->getCalledFunction()) {
    if (F->isIntrinsic()) {
      return false;
    }
  }

  return CI1->getNumArgOperands() == CI2->getNumArgOperands()
      && CI1->getCallingConv() == CI2->getCallingConv()
      && CI1->getAttributes() == CI2->getAttributes();
}

bool matchInstructions(Instruction *I1, Instruction *I2, const FunctionMergingOptions &Options) {
  if (I1->getOpcode()!=I2->getOpcode()) return false;

  if (I1->getOpcode() == Instruction::Ret)
    return true;

  if (I1->getNumOperands() != I2->getNumOperands())
    return false;

  bool sameType = false;
  sameType = (I1->getType() == I2->getType());
  for (unsigned i = 0; i < I1->getNumOperands(); i++) {
    sameType = sameType &&
               (I1->getOperand(i)->getType() == I2->getOperand(i)->getType());
  }
  if (!sameType)
    return false;

  switch (I1->getOpcode()) {
  case Instruction::InsertValue: return matchInsertValueInsts(dyn_cast<InsertValueInst>(I1), dyn_cast<InsertValueInst>(I2));
  case Instruction::ExtractValue: return matchExtractValueInsts(dyn_cast<ExtractValueInst>(I1), dyn_cast<ExtractValueInst>(I2));
  case Instruction::Switch: return matchSwitchInsts(dyn_cast<SwitchInst>(I1), dyn_cast<SwitchInst>(I2));
  case Instruction::GetElementPtr: 
    return matchGetElementPtrInsts(dyn_cast<GetElementPtrInst>(I1), dyn_cast<GetElementPtrInst>(I2));
  case Instruction::Call: return matchCallInsts(dyn_cast<CallInst>(I1), dyn_cast<CallInst>(I2));
  default:
    return I1->isSameOperationAs(I2);
  }
}

bool matchBlockEntry(BasicBlock *BB1, BasicBlock *BB2) {
  if (BB1->isLandingPad() || BB2->isLandingPad()) {
    LandingPadInst *LP1 = BB1->getLandingPadInst();
    LandingPadInst *LP2 = BB2->getLandingPadInst();
    if (LP1 == nullptr || LP2 == nullptr)
      return false;
    if (!matchLandingPad(LP1, LP2)) return false;
  }
  return true;
}

bool validMergeTypes(Function *F1, Function *F2, const FunctionMergingOptions &Options) {
  bool EquivTypes = F1->getReturnType()==F2->getReturnType();
  if (!EquivTypes &&
      !F1->getReturnType()->isVoidTy() && !F2->getReturnType()->isVoidTy()) {
      return false;
  }
  return true;
}

static bool validMergePair(Function *F1, Function *F2) {
  //if (F1->hasAvailableExternallyLinkage() ||
  //    F2->hasAvailableExternallyLinkage()) return false;
  //
  //if (!F1->isDiscardableIfUnused() || F2->isDiscardableIfUnused()) return false;

  if (F1->hasLinkOnceLinkage() ||
      F2->hasLinkOnceLinkage()) return false;

  //if (!F1->getSection().equals(F2->getSection())) return false;
  //if (F1->hasSection()!=F2->hasSection()) return false;
  //if (F1->hasSection() && !F1->getSection().equals(F2->getSection())) return false;

  //if (F1->hasComdat()!=F2->hasComdat()) return false;
  //if (F1->hasComdat() && F1->getComdat() != F2->getComdat()) return false;
  
  if (F1->hasPersonalityFn()!=F2->hasPersonalityFn()) return false;
  if (F1->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 != PersonalityFn2) return false;
  }
  
  return true;
}

static void MergeArguments(LLVMContext &Context, Function *F1, Function *F2,
  CodeAlignment *AR,
  std::map<unsigned, unsigned> &ParamMap1, std::map<unsigned, unsigned> &ParamMap2, std::vector<Type *> &Args, const FunctionMergingOptions &Options) {

  std::vector<Argument *> ArgsList1;
  for (Argument &arg : F1->args()) {
    ArgsList1.push_back(&arg);
  }

  Args.push_back(IntegerType::get(Context, 1)); // push the function Id argument
  unsigned ArgId = 0;
  for (auto I = F1->arg_begin(), E = F1->arg_end(); I != E; I++) {
    ParamMap1[ArgId] = Args.size();
    Args.push_back((*I).getType());
    ArgId++;
  }

  auto AttrList1 = F1->getAttributes();
  auto AttrList2 = F2->getAttributes();

  // merge arguments from Function2 with Function1
  ArgId = 0;
  for (auto I = F2->arg_begin(), E = F2->arg_end(); I != E; I++) {

    std::map<unsigned, int> MatchingScore;
    // first try to find an argument with the same name/type
    // otherwise try to match by type only
    for (unsigned i = 0; i < ArgsList1.size(); i++) {
      if (ArgsList1[i]->getType() == (*I).getType()) {
	
	auto AttrSet1 = AttrList1.getParamAttributes( ArgsList1[i]->getArgNo() );
	auto AttrSet2 = AttrList2.getParamAttributes( (*I).getArgNo() );
	if (AttrSet1!=AttrSet2) continue;

        // check for conflict from a previous matching
        bool hasConflict = false; 
        for (auto ParamPair : ParamMap2) {
          if (ParamPair.second == ParamMap1[i]) {
            hasConflict = true;
            break;
          }
        }
        if (hasConflict)
          continue;
        MatchingScore[i] = 0;
        if (!Options.MaximizeParamScore)
          break; // if not maximize score, get the first one
      }
    }

    
    if (MatchingScore.size() > 0) { // maximize scores
      for (auto &MB : AR->AlignedBlocks) {
        BasicBlock *BB1 = MB[0];
        BasicBlock *BB2 = MB[1];
	if (BB1==nullptr || BB2==nullptr) continue;

        auto It1 = BB1->begin();
        auto It2 = BB2->begin();

        while(isa<PHINode>(*It1) || isa<LandingPadInst>(*It1)) It1++;
        while(isa<PHINode>(*It2) || isa<LandingPadInst>(*It2)) It2++;

        while (It1!=BB1->end() && It2!=BB2->end()) {
          Instruction *I1 = &*It1;
          Instruction *I2 = &*It2;

	  if (MB.isMatchingPair(I1,I2)) {
            for (unsigned i = 0; i < I1->getNumOperands(); i++) {
              for (auto KV : MatchingScore) {
                if (I1->getOperand(i) == ArgsList1[KV.first]) {
                  if (i < I2->getNumOperands() && I2->getOperand(i) == &(*I)) {
                    MatchingScore[KV.first]++;
                  }
                }
              }
            }
          }
          It1++;
          It2++;
        }
      }

      int MaxScore = -1;
      int MaxId = 0;

      for (auto KV : MatchingScore) {
        if (KV.second > MaxScore) {
          MaxScore = KV.second;
          MaxId = KV.first;
        }
      }

      ParamMap2[ArgId] = ParamMap1[MaxId];
    } else {
      ParamMap2[ArgId] = Args.size();
      Args.push_back((*I).getType());
    }

    ArgId++;
  }

}

static void SetFunctionAttributes(Function *F1, Function *F2, Function *MergedFunc) {
  unsigned MaxAlignment = std::max(F1->getAlignment(), F2->getAlignment());
  if (F1->getAlignment()!=F2->getAlignment()) {
    errs() << "WARNING: different function alignment!\n";
  }
  if (MaxAlignment) MergedFunc->setAlignment(Align(MaxAlignment));

  if (F1->getCallingConv() == F2->getCallingConv()) {
    MergedFunc->setCallingConv(F1->getCallingConv());
  } else {
    errs() << "WARNING: different calling convention!\n";
    //MergedFunc->setCallingConv(CallingConv::Fast);
  }

/*
  if (F1->getLinkage() == F2->getLinkage()) {
    MergedFunc->setLinkage(F1->getLinkage());
  } else {
    if (Debug) errs() << "ERROR: different linkage type!\n";
    MergedFunc->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  }
*/


  /*
  if (F1->isDSOLocal() == F2->isDSOLocal()) {
    MergedFunc->setDSOLocal(F1->isDSOLocal());
  } else {
    if (Debug) errs() << "ERROR: different DSO local!\n";
  }
  */
  MergedFunc->setDSOLocal(true);

  if (F1->getSubprogram() == F2->getSubprogram()) {
    MergedFunc->setSubprogram(F1->getSubprogram());
  } else {
    errs() << "WARNING: different subprograms!\n";
  }

/*
  if (F1->getUnnamedAddr() == F2->getUnnamedAddr()) {
    MergedFunc->setUnnamedAddr(F1->getUnnamedAddr());
  } else {
    if (Debug) errs() << "ERROR: different unnamed addr!\n";
    MergedFunc->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
  }
*/
  MergedFunc->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  /*
  if (F1->getVisibility() == F2->getVisibility()) {
    //MergedFunc->setVisibility(F1->getVisibility());
  } else if (Debug) {
    errs() << "ERROR: different visibility!\n";
  }
  */
  MergedFunc->setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);

  // Exception Handling requires landing pads to have the same personality
  // function
  if (F1->hasPersonalityFn() && F2->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 == PersonalityFn2) {
      MergedFunc->setPersonalityFn(PersonalityFn1);
    } else {
      //errs() << "ERROR: different personality function!\n";
      errs() << "WARNING: different personality function!\n";
    }
  } else if (F1->hasPersonalityFn()) {
    //errs() << "Only F1 has PersonalityFn\n";
    // TODO: check if this is valid: merge function with personality with function without it
    MergedFunc->setPersonalityFn(F1->getPersonalityFn());
    errs() << "WARNING: only one personality function!\n";
  } else if (F2->hasPersonalityFn()) {
    //errs() << "Only F2 has PersonalityFn\n";
    // TODO: check if this is valid: merge function with personality with function without it
    MergedFunc->setPersonalityFn(F2->getPersonalityFn());
    errs() << "WARNING: only one personality function!\n";
  }

  if (F1->hasComdat() && F2->hasComdat()) {
    auto *Comdat1 = F1->getComdat();
    auto *Comdat2 = F2->getComdat();
    if (Comdat1 == Comdat2) {
      MergedFunc->setComdat(Comdat1);
    } else {
      errs() << "WARNING: different comdats!\n";
    }
  } else if (F1->hasComdat()) {
    //errs() << "Only F1 has Comdat\n";
    MergedFunc->setComdat(F1->getComdat()); // TODO: check if this is valid:
                                            // merge function with comdat with
                                            // function without it
    errs() << "WARNING: only one comdat!\n";
  } else if (F2->hasComdat()) {
    //errs() << "Only F2 has Comdat\n";
    MergedFunc->setComdat(F2->getComdat()); // TODO: check if this is valid:
                                            // merge function with comdat with
                                            // function without it
    errs() << "WARNING: only one comdat!\n";
  }

  if (F1->hasSection()) {
    MergedFunc->setSection(F1->getSection());
  }

}

void CodeMerger::destroyGeneratedCode() {
  for (Instruction *I : CreatedInsts) {
    I->dropAllReferences();
  }
  for (Instruction *I : CreatedInsts) {
    I->eraseFromParent();
  }
  for (BasicBlock *BB : CreatedBBs) {
    BB->eraseFromParent();
  }
  CreatedInsts.clear();
  CreatedBBs.clear();
}


class Fingerprint {
public:
  static const size_t MaxOpcode = 68;
  int OpcodeFreq[MaxOpcode];
  Function *F;
  BasicBlock *BB;

  Fingerprint() : F(nullptr), BB(nullptr) {}

  Fingerprint(Function *F) : F(F), BB(nullptr) {
    memset(OpcodeFreq, 0, sizeof(int) * MaxOpcode);
    for (Instruction &I : instructions(F)) {
      OpcodeFreq[I.getOpcode()]++;
    }
  }

  Fingerprint(BasicBlock *BB) : F(BB->getParent()), BB(BB) {
    memset(OpcodeFreq, 0, sizeof(int) * MaxOpcode);
    for (Instruction &I : *BB) {
      OpcodeFreq[I.getOpcode()]++;
    }
  }

  float magnitude() {
    unsigned Sum = 0;
    for (unsigned i = 0; i < MaxOpcode; i++) {
      auto Val = OpcodeFreq[i];
      Sum += Val*Val;
    }
    float Sqrt = std::sqrt((float)Sum);
    return Sqrt;
  }

  class Distances {
  public:
    static int manhattan(Fingerprint *FP1, Fingerprint *FP2) {
      int Distance = 0;
      for (size_t i = 0; i < Fingerprint::MaxOpcode; i++) {
        int Freq1 = FP1->OpcodeFreq[i];
        int Freq2 = FP2->OpcodeFreq[i];
        Distance += std::abs(Freq1-Freq2);
      }
      return Distance;
    }

    static float euclidean(Fingerprint *FP1, Fingerprint *FP2) {
      int Sum = 0;
      for (size_t i = 0; i < Fingerprint::MaxOpcode; i++) {
        int Freq1 = FP1->OpcodeFreq[i];
        int Freq2 = FP2->OpcodeFreq[i];
        int Sub = Freq1-Freq2;
        Sum += Sub*Sub;
      }
      float Distance = std::sqrt((float)Sum);
      return Distance;
    }

    static float cosine(Fingerprint *FP1, Fingerprint *FP2) {
      int AB = 0;
      int A2 = 0;
      int B2 = 0;
      for (size_t i = 0; i < Fingerprint::MaxOpcode; i++) {
        int Freq1 = FP1->OpcodeFreq[i];
        int Freq2 = FP2->OpcodeFreq[i];
        AB += Freq1*Freq2;
        A2 += Freq1*Freq1;
        B2 += Freq2*Freq2;
      }
      float Similarity = ((float)AB)/(std::sqrt((float)A2)*std::sqrt((float)B2));
      float Distance = 1.f - Similarity;
      return Distance;
    }
  };
};

class BlockFingerprint : public Fingerprint {
public:
  size_t Size;

  BlockFingerprint() : Size(0) {}

  BlockFingerprint(BasicBlock *BB) : Fingerprint(BB) {
    Size = 0;
    for (Instruction &I : *BB) {
      if (!isa<LandingPadInst>(&I) && !isa<PHINode>(&I)) {
        Size++;
      }
    }
  }

  int distance(BlockFingerprint &BF) {
    return Fingerprint::Distances::manhattan(this, &BF);
  }

};

class FunctionData {
public:
  Function *F;
  Fingerprint *FP;
  size_t Size;
  int Distance;
  std::list<FunctionData>::iterator iterator;

  FunctionData() : F(nullptr), FP(nullptr), Size(0), Distance(0) {}
  FunctionData(Function *F, Fingerprint *FP, size_t Size) : F(F), FP(FP), Size(Size), Distance(0) {}
};

class BlockData {
public:
  BasicBlock *BB;
  size_t Size;
  int Encoding;

  BlockData() : BB(nullptr), Size(0), Encoding(0) {}

  BlockData(BasicBlock *BB) : BB(BB) {
    Size = 0;
    for (Instruction &I : *BB) {
      if (!isa<LandingPadInst>(&I) && !isa<PHINode>(&I)) {
        Size++;
        Encoding += I.getOpcode();
      } else if (isa<LandingPadInst>(&I)) Encoding += I.getOpcode();
    }
  }
};

PairwiseAlignment::PairwiseAlignment(ArrayRef<BasicBlock*> Blocks1, ArrayRef<BasicBlock*> Blocks2, const FunctionMergingOptions &Options) {
  int TotalMatches = 0;
  int TotalInsts = 0;
  int TotalCoreMatches = 0;

  std::map<size_t, std::vector<BlockFingerprint> > BlocksF1;
  
  for (BasicBlock *BB1 : Blocks1) {
    BlockFingerprint BD1(BB1);
    BlocksF1[BD1.Size].push_back(BD1);
  }
  
  for (BasicBlock *BB2 : Blocks2) {
    BlockFingerprint BD2(BB2);
    
    auto &SetRef = BlocksF1[BD2.Size];

    auto BestIt = SetRef.end();
    int BestDist = std::numeric_limits<int>::max();
    for (auto BDIt = SetRef.begin(), E = SetRef.end(); BDIt!=E; BDIt++) {
      auto D = BD2.distance(*BDIt);
      if (D<BestDist) {
        BestDist = D;
        BestIt = BDIt;
      }
    }
    if (BestIt!=SetRef.end()) {
      auto &BD1 = *BestIt;
      BasicBlock *BB1 = BD1.BB;
      
        auto It1 = BB1->begin();
        auto It2 = BB2->begin();

        //Analyse block profitability
        int OriginalCost = 0;
        int MergedCost = 0;

        bool InsideSplit = false;
        if (matchBlockEntry(BB1,BB2)) {
          InsideSplit = false;
        } else {
          InsideSplit = true;
          MergedCost += 1;
        }

        while(isa<PHINode>(*It1) || isa<LandingPadInst>(*It1)) It1++;
        while(isa<PHINode>(*It2) || isa<LandingPadInst>(*It2)) It2++;

        while (It1!=BB1->end() && It2!=BB2->end()) {
          Instruction *I1 = &*It1;
          Instruction *I2 = &*It2;
            
          OriginalCost += 2;

          if (matchInstructions(I1,I2,Options)) {
            MergedCost += 1; //reduces 1 inst by merging two insts into one
            if (InsideSplit) {
              InsideSplit = false;
              MergedCost += 2; //two branches to converge
            }

          } else {
            if (!InsideSplit) {
              InsideSplit = true;
              MergedCost += 1; //one branch to split
            }
            MergedCost += 2; //two instructions
          }
          It1++;
          It2++;
        }

        if (It1!=BB1->end() || It2!=BB2->end()) {
          //return false; ERROR
          CodeAlignment::AlignedBlocks.clear();
	  return;
        }

        It1 = BB1->begin();
        It2 = BB2->begin();

        bool Profitable = (MergedCost <= OriginalCost);
        if (Profitable) {
	  CodeAlignment::AlignedBlocks.push_back(MatchingBlocks(BB1,BB2));
          auto &MB = CodeAlignment::AlignedBlocks.back();

          while(isa<PHINode>(*It1) || isa<LandingPadInst>(*It1)) It1++;
          while(isa<PHINode>(*It2) || isa<LandingPadInst>(*It2)) It2++;

          while (It1!=BB1->end() && It2!=BB2->end()) {
            Instruction *I1 = &*It1;
            Instruction *I2 = &*It2;

    	TotalInsts++;
            if (matchInstructions(I1,I2,Options)) {
    	  MB.setMatchingPair(I1,I2);
              TotalMatches++;
    	  if (!I1->isTerminator()) {
    	    TotalCoreMatches++;
    	  }
            }

            It1++;
            It2++;
          }

          if (It1!=BB1->end() || It2!=BB2->end()) {
            //return false; ERROR
            CodeAlignment::AlignedBlocks.clear();
	    return;
          }
          
          SetRef.erase(BestIt);
        }
    }
  }

  bool Profitable = (TotalMatches==TotalInsts) || (TotalCoreMatches>0);
  if (!Profitable) CodeAlignment::AlignedBlocks.clear();
}

FunctionMergeResult llvm::MergeFunctions(Function *F1, Function *F2, const char *Name, const FunctionMergingOptions &Options) {
  FunctionMergeResult ErrorResponse(F1, F2, nullptr);

  if (F1->getParent()!=F2->getParent())
    return ErrorResponse;

  Module *M = F1->getParent();

  if (!validMergePair(F1,F2))
    return ErrorResponse;

  SmallVector<BasicBlock*, 8> Blocks1;
  for (auto &BB : *F1) Blocks1.push_back(&BB);
  SmallVector<BasicBlock*, 8> Blocks2;
  for (auto &BB : *F2) Blocks2.push_back(&BB);

  PairwiseAlignment AR(Blocks1,Blocks2,Options);
  if (AR.AlignedBlocks.empty())
    return ErrorResponse;

  CodeMerger CG(M,Blocks1,Blocks2);

  ValueToValueMapTy VMap;

  CG.setEntryPoints(&F1->getEntryBlock(), &F2->getEntryBlock());

  FunctionMergeResult Result = CG.defineMergedFunction(F1,F2,Name,&AR,VMap,Options);
  if (!CG.generate(&AR, VMap, Options)) {
    CG.getFunction()->eraseFromParent();
    return ErrorResponse;
  }

  /*
  if (!RequiresFuncId) {
    errs() << "Removing FuncId\n";
    
    MergedFunc = RemoveFuncIdArg(MergedFunc, ArgsList);

    for (auto &kv : ParamMap1) {
      ParamMap1[kv.first] = kv.second - 1;
    }
    for (auto &kv : ParamMap2) {
      ParamMap2[kv.first] = kv.second - 1;
    }
    FuncId = nullptr;
    
  }
  */

  //Result.setFunctionIdArgument(FuncId != nullptr);
  return Result;
  
}

void llvm::ReplaceFunctionByCall(Function *F, FunctionMergeResult &MFR) {
  LLVMContext &Context = F->getParent()->getContext();
  const DataLayout *DL = &F->getParent()->getDataLayout();

  Value *FuncId = MFR.getFunctionIdValue(F);
  Function *MergedF = MFR.getMergedFunction();

  F->deleteBody();
  BasicBlock *NewBB = BasicBlock::Create(Context, "", F);
  IRBuilder<> Builder(NewBB);

  std::vector<Value *> args;
  for (unsigned i = 0; i < MergedF->getFunctionType()->getNumParams(); i++) {
    args.push_back(nullptr);
  }

  if (MFR.hasFunctionIdArgument()) {
    args[0] = FuncId;
  }

  std::vector<Argument *> ArgsList;
  for (Argument &arg : F->args()) {
    ArgsList.push_back(&arg);
  }

  for (auto Pair : MFR.getArgumentMapping(F)) {
    args[Pair.second] = ArgsList[Pair.first];
  }

  for (unsigned i = 0; i < args.size(); i++) {
    if (args[i] == nullptr) {
      args[i] = UndefValue::get(MergedF->getFunctionType()->getParamType(i));
    }
  }

  CallInst *CI =
      (CallInst *)Builder.CreateCall(MergedF, ArrayRef<Value *>(args));
  CI->setTailCall();
  CI->setCallingConv(MergedF->getCallingConv());
  CI->setAttributes(MergedF->getAttributes());
  CI->setIsNoInline();

  if (F->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Value *CastedV = CI;
    if (MFR.needUnifiedReturn()) {
      Value *AddrCI = Builder.CreateAlloca(CI->getType());
      Builder.CreateStore(CI,AddrCI);
      Value *CastedAddr = Builder.CreatePointerCast(AddrCI, PointerType::get(F->getReturnType(), DL->getAllocaAddrSpace()));
      CastedV = Builder.CreateLoad(F->getReturnType(), CastedAddr);
    }
    Builder.CreateRet(CastedV);
  }
}

bool llvm::ReplaceCallsWith(Function *F, FunctionMergeResult &MFR) {
  const DataLayout *DL = &F->getParent()->getDataLayout();

  Value *FuncId = MFR.getFunctionIdValue(F);
  Function *MergedF = MFR.getMergedFunction();

  unsigned CountUsers = 0;
  std::vector<CallBase *> Calls;
  for (User *U : F->users()) {
    CountUsers++;
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getCalledFunction() == F) {
        Calls.push_back(CI);
      }
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(U)) {
      if (II->getCalledFunction() == F) {
        Calls.push_back(II);
      }
    }
  }

  if (Calls.size()<CountUsers)
    return false;
  
  for (CallBase *CI : Calls) {
    IRBuilder<> Builder(CI);

    std::vector<Value *> args;
    for (unsigned i = 0; i < MergedF->getFunctionType()->getNumParams(); i++) {
      args.push_back(nullptr);
    }

    if (MFR.hasFunctionIdArgument()) {
      args[0] = FuncId;
    }

    for (auto Pair : MFR.getArgumentMapping(F)) {
      args[Pair.second] = CI->getArgOperand(Pair.first);
    }

    for (unsigned i = 0; i < args.size(); i++) {
      if (args[i] == nullptr) {
        args[i] = UndefValue::get(MergedF->getFunctionType()->getParamType(i));
      }
    }

    CallBase *NewCB = nullptr;
    if (CI->getOpcode()==Instruction::Call) {
      NewCB = (CallInst *)Builder.CreateCall(MergedF->getFunctionType(),
                                                      MergedF, args);
    } else if (CI->getOpcode()==Instruction::Invoke) {
      InvokeInst *II = dyn_cast<InvokeInst>(CI);
      NewCB = (InvokeInst *)Builder.CreateInvoke(MergedF->getFunctionType(),
                                                      MergedF, II->getNormalDest(), II->getUnwindDest(), args);
    }
    NewCB->setCallingConv(MergedF->getCallingConv());
    NewCB->setAttributes(MergedF->getAttributes());
    NewCB->setIsNoInline();
    Value *CastedV = NewCB;
    if (!F->getReturnType()->isVoidTy()) {
      if (MFR.needUnifiedReturn()) {
        Value *AddrCI = Builder.CreateAlloca(NewCB->getType());
        Builder.CreateStore(NewCB,AddrCI);
        Value *CastedAddr = Builder.CreatePointerCast(AddrCI, PointerType::get(F->getReturnType(), DL->getAllocaAddrSpace()));
        CastedV = Builder.CreateLoad(F->getReturnType(), CastedAddr);
      }
    }

    if (CI->getNumUses() > 0) {
      CI->replaceAllUsesWith(CastedV);
    }
    CI->eraseFromParent();
  }

  return true;
}

static bool ShouldPreserveGV(const GlobalValue *GV) {
  // Function must be defined here
  if (GV->isDeclaration())
    return true;

  // Available externally is really just a "declaration with a body".
  //if (GV->hasAvailableExternallyLinkage())
  //  return true;

  // Assume that dllexported symbols are referenced elsewhere
  if (GV->hasDLLExportStorageClass())
    return true;

  return false;
}

static int RequiresOriginalInterface(Function *F, FunctionMergeResult &MFR, StringSet<> &AlwaysPreserved) {
  bool CanErase = !F->hasAddressTaken();
  CanErase = CanErase && (AlwaysPreserved.find(F->getName())==AlwaysPreserved.end());
  CanErase = CanErase && F->isDiscardableIfUnused();
  return !CanErase;
}

static int RequiresOriginalInterfaces(FunctionMergeResult &MFR, StringSet<> &AlwaysPreserved) {
  auto FPair = MFR.getFunctions();
  Function *F1 = FPair.first;
  Function *F2 = FPair.second;
  return (RequiresOriginalInterface(F1, MFR, AlwaysPreserved)?1:0) +
         (RequiresOriginalInterface(F2, MFR, AlwaysPreserved)?1:0);
}

void UpdateCallGraph(Function *F, FunctionMergeResult &MFR, StringSet<> &AlwaysPreserved) {
  ReplaceFunctionByCall(F, MFR);
  if (!RequiresOriginalInterface(F,MFR, AlwaysPreserved)) {
    bool CanErase = ReplaceCallsWith(F, MFR);
    CanErase = CanErase && F->use_empty();
    CanErase = CanErase && (AlwaysPreserved.find(F->getName())==AlwaysPreserved.end());
    CanErase = CanErase && !ShouldPreserveGV(F);
    CanErase = CanErase && F->isDiscardableIfUnused();
    if (CanErase) F->eraseFromParent();
  }
}

void UpdateCallGraph(FunctionMergeResult &MFR, StringSet<> &AlwaysPreserved) {
  auto FPair = MFR.getFunctions();
  Function *F1 = FPair.first;
  Function *F2 = FPair.second;
  UpdateCallGraph(F1, MFR, AlwaysPreserved);
  UpdateCallGraph(F2, MFR, AlwaysPreserved);
}

static int EstimateThunkOverhead(FunctionMergeResult &MFR, StringSet<> &AlwaysPreserved) {
  return RequiresOriginalInterfaces(MFR, AlwaysPreserved)*(2+MFR.getMergedFunction()->getFunctionType()->getNumParams());
}

size_t EstimateFunctionSize(Function *F, TargetTransformInfo *TTI) {
  InstructionCost size = 0;
  for (Instruction &I : instructions(F)) {
    size += TTI->getInstructionCost(
      &I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
  }

  auto OptSize = size.getValue();
  if (OptSize.hasValue()) return OptSize.getValue();
  else return std::numeric_limits<size_t>::max();
}

bool FunctionMerging::runImpl(Module &M) {

  StringSet<> AlwaysPreserved;
  AlwaysPreserved.insert("main");

  FunctionMergingOptions Options = FunctionMergingOptions()
                                    .maximizeParameterScore(MaxParamScore)
                                    .enableUnifiedReturnTypes(EnableUnifiedReturnType);

  //TODO: We could use a TTI pass instead.
  TargetTransformInfo TTI(M.getDataLayout());

  std::vector<FunctionData> FunctionsToProcess;

  for (auto &F : M) {
    if (F.isDeclaration() || F.isVarArg() || F.hasAvailableExternallyLinkage())
      continue;
    
    FunctionData FD(&F, new Fingerprint(&F), EstimateFunctionSize(&F, &TTI));
    FunctionsToProcess.push_back(FD);
  }

  
  std::sort(FunctionsToProcess.begin(), FunctionsToProcess.end(),
	    [](auto &F1, auto &F2) {
	      return F1.Size > F2.Size;
	    });

  //sort fingerprints by magnitude
  std::stable_sort(FunctionsToProcess.begin(), FunctionsToProcess.end(),
      [&](auto &FD1, auto &FD2) -> bool {
        return FD1.FP->magnitude() > FD2.FP->magnitude();
  });

  std::list<FunctionData> WorkList;

  for (auto &FD : FunctionsToProcess) {
    WorkList.push_back(FD);
  }

  FunctionsToProcess.clear();

  while (!WorkList.empty()) {
    FunctionData FD1 = WorkList.front();
    WorkList.pop_front();

    Function *F1 = FD1.F;

    bool FoundCandidate = false;
    FunctionData BestFD2;
    int BestDist = std::numeric_limits<int>::max();

    unsigned CountCandidates = 0;
    for (auto It = WorkList.begin(), E = WorkList.end(); It!=E; It++) {
      FunctionData &FD2 = *It;
      Function *F2 = FD2.F;

      if ((!validMergeTypes(F1, F2, Options) && !Options.EnableUnifiedReturnType) || !validMergePair(F1, F2))
        continue;

      FD2.iterator = It;
      auto Dist = Fingerprint::Distances::manhattan(FD1.FP, FD2.FP);
      FD2.Distance = Dist;
      if (Dist < BestDist) {
        BestDist = Dist;
        BestFD2 = FD2;
        FoundCandidate = true;
      }
      if (RankingThreshold && CountCandidates>RankingThreshold) {
        break;
      }
      CountCandidates++;
    }
    if (!FoundCandidate) continue;

    delete FD1.FP;
    FD1.FP = nullptr;

    Function *F2 = BestFD2.F;

    LLVM_DEBUG(dbgs() << "Attempting: " << F1->getName() << ", " << F2->getName() << "\n");

    FunctionMergeResult Result = MergeFunctions(F1,F2, "mf",Options);
    
    if (Result.getMergedFunction()) {

      int SizeF12 = EstimateThunkOverhead(Result, AlwaysPreserved) +
                    EstimateFunctionSize(Result.getMergedFunction(), &TTI);

      int SizeF1F2 = FD1.Size+BestFD2.Size;

      int PercentReduction = 100*(double)(SizeF1F2 - SizeF12)/SizeF1F2;
      //bool Profitable = (SizeF12 + MergingOverheadThreshold) < SizeF1F2;
      bool Profitable = PercentReduction > MergingOverheadThreshold;

      LLVM_DEBUG(dbgs() << "Estimated Reduction: " << PercentReduction
             << "% (" << Profitable << ")"
             << " : " << F1->getName()
             << ", " << F2->getName() << "\n");

      if (Profitable || ForceMerge) {
        
        LLVM_DEBUG(dbgs() << "Merged: " << F1->getName() << ", " << F2->getName()
               << " = " << Result.getMergedFunction()->getName() << "\n");
        
        UpdateCallGraph(Result, AlwaysPreserved);

        ++NumMergedFunctions;

        WorkList.erase(BestFD2.iterator);
        delete BestFD2.FP;
        BestFD2.FP = nullptr;

        if (ReuseMergedFunctions) {
          // feed new function back into the working lists
          FunctionData MFD(Result.getMergedFunction(),
                          new Fingerprint(Result.getMergedFunction()),
      		    EstimateFunctionSize(Result.getMergedFunction(), &TTI));
          WorkList.push_front(MFD);
        }

      } else {
        if (Result.getMergedFunction() != nullptr)
          Result.getMergedFunction()->eraseFromParent();
      }
    }
  }

  return true;
}

PreservedAnalyses FunctionMergingPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  FunctionMerging FM;
  if (!FM.runImpl(M))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

class FunctionMergingLegacyPass : public ModulePass {
public:
  static char ID;
  FunctionMergingLegacyPass() : ModulePass(ID) {
     initializeFunctionMergingLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override {
    FunctionMerging FM;
    return FM.runImpl(M);
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};

char FunctionMergingLegacyPass::ID = 0;
INITIALIZE_PASS(FunctionMergingLegacyPass, "func-merging", "New Function Merging", false,
                false)

ModulePass *llvm::createFunctionMergingPass() {
  return new FunctionMergingLegacyPass();
}

Instruction *CodeMerger::cloneInst(IRBuilder<> &Builder, Function *MF, Instruction *I) {
  Instruction *NewI = nullptr;
  if (I->getOpcode() == Instruction::Ret) {
    if (MF->getReturnType()->isVoidTy()) {
      NewI = Builder.CreateRetVoid();
    } else {
      NewI = Builder.CreateRet(
          UndefValue::get(MF->getReturnType()));
    }
  } else {
    NewI = I->clone();
    for(unsigned i = 0; i<NewI->getNumOperands(); i++) {
      if (!isa<Constant>(I->getOperand(i)))
        NewI->setOperand(i,nullptr);
    }
    Builder.Insert(NewI);

  }
  
  // TODO: merge metadata
  // currently clearing metadata
  SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
  NewI->getAllMetadata(MDs);
  for (std::pair<unsigned, MDNode *> MDPair : MDs) {
    NewI->setMetadata(MDPair.first, nullptr);
  }
  
  return NewI;
}

void CodeMerger::codeGen(CodeAlignment *AR, ValueToValueMapTy &VMap,
                    std::unordered_map<BasicBlock *, BasicBlock *> &BlocksF1,
                    std::unordered_map<BasicBlock *, BasicBlock *> &BlocksF2,
                    std::list<MergedInstruction> &MergedInsts) {

  Function *MergedFunc = getFunction();
  auto Blocks1 = getBlocks1();
  auto Blocks2 = getBlocks2();
  BasicBlock *EntryBB1 = getEntryBlock1();
  BasicBlock *EntryBB2 = getEntryBlock2();
  BasicBlock *PreBB = getPreBlock();
  Value *IsFunc1 = getCondition();

  std::unordered_set<BasicBlock *> BlocksCloned;
  
  //errs() << "Cloning Merged Blocks \n";
  for (auto &MB : AR->AlignedBlocks) {
    BasicBlock *BB1 = MB[0];
    BasicBlock *BB2 = MB[1];

    if (BB1 && BB2) {
      BlocksCloned.insert(BB1);
      BlocksCloned.insert(BB2);

      BasicBlock *MergedBB = BasicBlock::Create(MergedFunc->getContext(), "merged.bb", MergedFunc);

      VMap[BB1] = MergedBB;
      VMap[BB2] = MergedBB;
      
      BlocksF1[MergedBB] = BB1;
      BlocksF2[MergedBB] = BB2;

      IRBuilder<> Builder(MergedBB);

      auto It1 = BB1->begin();

      while( It1!=BB1->end() && (isa<PHINode>(*It1) || isa<LandingPadInst>(*It1)) ) {
        if (isa<PHINode>(*It1)) {
          auto *PHI = Builder.CreatePHI((*It1).getType(), 0);
          VMap[&(*It1)] = PHI;
          MergedInsts.push_back(MergedInstruction(PHI,&(*It1),nullptr));
        }
        It1++;
      }

      auto It2 = BB2->begin();
      while( It2!=BB2->end() && (isa<PHINode>(*It2) || isa<LandingPadInst>(*It2)) ) {
        if (isa<PHINode>(*It2)) {
          auto *PHI = Builder.CreatePHI((*It2).getType(), 0);
          VMap[&(*It2)] = PHI;
          MergedInsts.push_back(MergedInstruction(PHI,nullptr,&(*It2)));
        }
        It2++;
      }

      BasicBlock *SplitBB1 = nullptr;
      BasicBlock *SplitBB2 = nullptr;
      while (It1!=BB1->end() || It2!=BB2->end()) {
        Instruction *I1 = (It1!=BB1->end())? &*It1 : nullptr;
        Instruction *I2 = (It2!=BB2->end())? &*It2 : nullptr;

	Instruction *MappedI2 = MB.getMatchingInstruction(I1);
	if (MappedI2==I2) {
          if (MergedBB==nullptr) {
            MergedBB = BasicBlock::Create(MergedFunc->getContext(), "merged.bb", MergedFunc);
            BlocksF1[MergedBB] = BB1;
            BlocksF2[MergedBB] = BB2;

	    Builder.SetInsertPoint(SplitBB1);
	    Builder.CreateBr(MergedBB);
	    Builder.SetInsertPoint(SplitBB2);
	    Builder.CreateBr(MergedBB);
	  }
	  Builder.SetInsertPoint(MergedBB);
          Instruction *NewI = cloneInst(Builder,MergedFunc,I1);

          VMap[I1] = NewI;
          VMap[I2] = NewI;
          MergedInsts.push_back(MergedInstruction(NewI,I1,I2));

          It1++;
          It2++;
	} else {
          if (MergedBB) {
	    //split
            SplitBB1 = BasicBlock::Create(MergedFunc->getContext(), "split.1.bb", MergedFunc);
            SplitBB2 = BasicBlock::Create(MergedFunc->getContext(), "split.2.bb", MergedFunc);
            BlocksF1[SplitBB1] = BB1;
            BlocksF2[SplitBB2] = BB2;

	    Builder.SetInsertPoint(MergedBB);
	    Builder.CreateCondBr(IsFunc1, SplitBB1, SplitBB2);
	    MergedBB = nullptr;
	  }

	  if (MappedI2==nullptr && I1!=nullptr) {
	    Builder.SetInsertPoint(SplitBB1);
            Instruction *NewI = cloneInst(Builder,MergedFunc,I1);
            VMap[I1] = NewI;
            MergedInsts.push_back(MergedInstruction(NewI,I1,nullptr));
            It1++;
	  } else {
            Builder.SetInsertPoint(SplitBB2);
            Instruction *NewI = cloneInst(Builder,MergedFunc,I2);
            VMap[I2] = NewI;
            MergedInsts.push_back(MergedInstruction(NewI,nullptr,I2));
            It2++;
	  }
	}
      }
    }
  }

  //errs() << "Cloning Blocks Function 1 \n";
  for (BasicBlock *BB1 : Blocks1) {

    if (BlocksCloned.find(BB1)==BlocksCloned.end()) {
      BlocksCloned.insert(BB1);

      std::string BBName = std::string("bb1.")+BB1->getName().str();
      BasicBlock *NewBB = BasicBlock::Create(MergedFunc->getContext(), BBName, MergedFunc);
      BlocksF1[NewBB] = BB1;
      VMap[BB1] = NewBB;
      
      IRBuilder<> Builder(NewBB);
      for (Instruction &I : *BB1) {
        if (isa<LandingPadInst>(&I))
          continue;
        if (isa<PHINode>(&I)) {
          auto *PHI = Builder.CreatePHI(I.getType(), 0);
          VMap[&I] = PHI;
          MergedInsts.push_back(MergedInstruction(PHI,&I,nullptr));
        } else {
          Instruction *NewI = cloneInst(Builder,MergedFunc,&I);
          VMap[&I] = NewI;
          MergedInsts.push_back(MergedInstruction(NewI,&I,nullptr));
	}
      }
    }
  }

  //errs() << "Cloning Blocks Function 2 \n";
  for (BasicBlock *BB2 : Blocks2) {

    if (BlocksCloned.find(BB2)==BlocksCloned.end()) {
      BlocksCloned.insert(BB2);

      std::string BBName = std::string("bb2.")+BB2->getName().str();
      BasicBlock *NewBB = BasicBlock::Create(MergedFunc->getContext(), BBName, MergedFunc);
      BlocksF2[NewBB] = BB2;
      VMap[BB2] = NewBB;
      
      IRBuilder<> Builder(NewBB);
      for (Instruction &I : *BB2) {
        if (isa<LandingPadInst>(&I))
          continue;
        if (isa<PHINode>(&I)) {
          auto *PHI = Builder.CreatePHI(I.getType(), 0);
          VMap[&I] = PHI;
          MergedInsts.push_back(MergedInstruction(PHI,nullptr,&I));
        } else {
          Instruction *NewI = cloneInst(Builder,MergedFunc,&I);
          VMap[&I] = NewI;
          MergedInsts.push_back(MergedInstruction(NewI,nullptr,&I));
	}
      }
    }
  }

  //wiring PreBB with entry block from each function
  BasicBlock *BB1 = dyn_cast<BasicBlock>(VMap[EntryBB1]);
  BasicBlock *BB2 = dyn_cast<BasicBlock>(VMap[EntryBB2]);

  BlocksF1[PreBB] = BB1;
  BlocksF2[PreBB] = BB2;

  IRBuilder<> Builder(PreBB);
  if (BB1 == BB2) {
    Builder.CreateBr(BB1);
  } else {
    Builder.CreateCondBr(IsFunc1, BB1, BB2);
  }
}

Value *CodeMerger::mergeValues(Value *V1, Value *V2, Instruction *InsertPt) {
  if (V1 == V2)
    return V1;

  LLVMContext &Context = getContext();
  Value *IsFirst = getCondition();

  if (V1 == ConstantInt::getTrue(Context) &&
      V2 == ConstantInt::getFalse(Context)) {
    return IsFirst;
  } else if (V1 == ConstantInt::getFalse(Context) &&
             V2 == ConstantInt::getTrue(Context)) {
    IRBuilder<> Builder(InsertPt);
    return Builder.CreateNot(IsFirst); /// TODO: create a single not(IsFunc1) for each merged function that needs it
  }

  IRBuilder<> Builder(InsertPt);
  Instruction *Sel = (Instruction *)Builder.CreateSelect(IsFirst, V1, V2);
  if (SelectInst *SelI = dyn_cast<SelectInst>(Sel)) {
    AllSelections.push_back(SelI);
  }
  return Sel;
}

bool CodeMerger::assignOperands(Instruction *I, bool IsFuncId1, ValueToValueMapTy &VMap) {
  Instruction *NewI = dyn_cast<Instruction>(VMap[I]);
  IRBuilder<> Builder(NewI);

  bool RequiresUnifiedReturn = getRequiresUnifiedReturn();
  Type *ReturnType = getReturnType();

  if (I->getOpcode() == Instruction::Ret && RequiresUnifiedReturn) {
    Value *V = MapValue(I->getOperand(0), VMap);
    assert( V!=nullptr && "Unexpected null value in operand assignment");

    if (V->getType() != ReturnType) {
      Value *Addr = Builder.CreateAlloca(V->getType());
      Builder.CreateStore(V, Addr);
      Value *CastedAddr = Builder.CreatePointerCast(Addr, PointerType::getUnqual(ReturnType));
      V = Builder.CreateLoad(ReturnType, CastedAddr);
    }
    NewI->setOperand(0, V);
  } else {
    for (unsigned i = 0; i < I->getNumOperands(); i++) {
      if (isa<BasicBlock>(I->getOperand(i)))
        continue;

      Value *V = MapValue(I->getOperand(i), VMap);
      assert( V!=nullptr && "Unexpected null value in operand assignment");

      NewI->setOperand(i, V);
    }
  }
    
  return true;
}

bool CodeMerger::assignLabelOperands(Instruction *I, std::unordered_map<BasicBlock*, BasicBlock *> &BlocksReMap, ValueToValueMapTy &VMap) {
  Instruction *NewI = dyn_cast<Instruction>(VMap[I]);

  LLVMContext &Context = getContext();
  Function *MergedFunc = getFunction();
  for (unsigned i = 0; i < I->getNumOperands(); i++) {
    //handling just label operands for now
    if (!isa<BasicBlock>(I->getOperand(i))) continue;
    BasicBlock *FXBB = dyn_cast<BasicBlock>(I->getOperand(i));

    Value *V = MapValue(FXBB, VMap);
    assert( V!=nullptr && "Unexpected null value in label operand");

    if (FXBB->isLandingPad()) {

      LandingPadInst *LP = FXBB->getLandingPadInst();

      BasicBlock *LPadBB = BasicBlock::Create(Context, "lpad.bb", MergedFunc);
      IRBuilder<> BuilderBB(LPadBB);
 
      Instruction *NewLP = LP->clone();
      BuilderBB.Insert(NewLP);
      VMap[LP] = NewLP;
      BlocksReMap[LPadBB] = I->getParent(); 
 
      BuilderBB.CreateBr(dyn_cast<BasicBlock>(V));
 
      V = LPadBB;
    }

    NewI->setOperand(i, V);
  }
  return true;
}

bool CodeMerger::assignPHIOperandsInBlock(BasicBlock *BB, std::unordered_map<BasicBlock*, BasicBlock *> &BlocksReMap, ValueToValueMapTy &VMap) {
  auto It = BB->begin();
  auto *EndI = BB->getFirstNonPHI();
  while (It!=BB->end() && (&*It)!=EndI) {
    Instruction *I = &*It;
    It++;
    if (PHINode *PHI = dyn_cast<PHINode>(I)) {
      PHINode *NewPHI = dyn_cast<PHINode>(VMap[PHI]);

      std::set<int> FoundIndices;
      std::set<BasicBlock*> IncomingBlocks;
      for (unsigned i = 0; i<PHI->getNumIncomingValues(); i++) IncomingBlocks.insert(PHI->getIncomingBlock(i));

      for (auto ItP = pred_begin(NewPHI->getParent()),
                E = pred_end(NewPHI->getParent());
           ItP != E; ItP++) {

        BasicBlock *NewPredBB = *ItP;

        Value *V = nullptr;

        if (BlocksReMap.find(NewPredBB)!=BlocksReMap.end()) {
          int Index = PHI->getBasicBlockIndex(BlocksReMap[NewPredBB]);
          if (Index>=0) {
            V = MapValue(PHI->getIncomingValue(Index), VMap);
            FoundIndices.insert(Index);
          }
        }

        if (V==nullptr) V = UndefValue::get(NewPHI->getType());

        NewPHI->addIncoming(V, NewPredBB);
      }
      if (FoundIndices.size()!=IncomingBlocks.size()) { //PHI->getNumIncomingValues()
	BB->dump();
	PHI->dump();
	NewPHI->dump();
        return false;
      }
    }
  }
  return true;
}

void CodeMerger::storeInstIntoAddr(Instruction *IV, Value *Addr) {
  IRBuilder<> Builder(IV->getParent());
  if (IV->isTerminator()) {
    BasicBlock *SrcBB = IV->getParent();
    if (InvokeInst *II = dyn_cast<InvokeInst>(IV)) {
      BasicBlock *DestBB = II->getNormalDest();

      Builder.SetInsertPoint(&*DestBB->getFirstInsertionPt());
      // create PHI
      PHINode *PHI = Builder.CreatePHI(IV->getType(), 0);
      for (auto PredIt = pred_begin(DestBB), PredE = pred_end(DestBB); PredIt != PredE; PredIt++) {
        BasicBlock *PredBB = *PredIt;
        if (PredBB == SrcBB) {
          PHI->addIncoming(IV, PredBB);
        } else {
          PHI->addIncoming(UndefValue::get(IV->getType()), PredBB);
        }
      }
      Builder.CreateStore(PHI, Addr);
    } else {
      for (auto SuccIt = succ_begin(SrcBB), SuccE = succ_end(SrcBB); SuccIt!=SuccE; SuccIt++) {
        BasicBlock *DestBB = *SuccIt;

        Builder.SetInsertPoint(&*DestBB->getFirstInsertionPt());
        // create PHI
        PHINode *PHI = Builder.CreatePHI(IV->getType(), 0);
        for (auto PredIt = pred_begin(DestBB), PredE = pred_end(DestBB); PredIt != PredE; PredIt++) {
          BasicBlock *PredBB = *PredIt;
          if (PredBB == SrcBB) {
            PHI->addIncoming(IV, PredBB);
          } else {
            PHI->addIncoming(UndefValue::get(IV->getType()), PredBB);
          }
        }
        Builder.CreateStore(PHI, Addr);
      }
    }
  } else {
    Instruction *LastI = nullptr;
    Instruction *InsertPt = nullptr;
    for (Instruction &I : *IV->getParent()) {
      InsertPt = &I;
      if (LastI == IV)
        break;
      LastI = &I;
    }
    if (isa<PHINode>(InsertPt) || isa<LandingPadInst>(InsertPt)) {
      Builder.SetInsertPoint(IV->getParent()->getTerminator());
    } else
      Builder.SetInsertPoint(InsertPt);

    Builder.CreateStore(IV, Addr);
  }
}

AllocaInst *CodeMerger::memfyInst(Instruction *I) {
  BasicBlock *PreBB = getPreBlock();

  IRBuilder<> Builder(&*PreBB->getFirstInsertionPt());
  Type *Ty = I->getType();
  AllocaInst *Addr = Builder.CreateAlloca(Ty);
  
  std::map<Value *, Value *> CachedLoads;
  for (auto UIt = I->use_begin(), E = I->use_end(); UIt != E;) {
    Use &UI = *UIt;
    UIt++;

    Instruction *User = cast<Instruction>(UI.getUser());

    if (PHINode *PHI = dyn_cast<PHINode>(User)) {
      auto *P = PHI->getIncomingBlock(UI.getOperandNo())->getTerminator();
      if (CachedLoads.find(P)==CachedLoads.end()) {
        IRBuilder<> Builder(P);
        auto *L = Builder.CreateLoad(Ty, Addr);
        UI.set(L);
        CachedLoads[P] = L;
      } else UI.set(CachedLoads[P]);
    } else {
      if (CachedLoads.find(User)==CachedLoads.end()) {
        IRBuilder<> Builder(User);
        auto *L = Builder.CreateLoad(Ty, Addr);
        UI.set(L);
        CachedLoads[User] = L;
      } else UI.set(CachedLoads[User]);
    }
  }

  storeInstIntoAddr(I, Addr);

  return Addr;
}

bool CodeMerger::defineReturnType(Function *F1, Function *F2, const FunctionMergingOptions &Options) {
  RetType1 = F1->getReturnType();
  RetType2 = F2->getReturnType();
  ReturnType = nullptr;
  RequiresUnifiedReturn = false;
  
  if (validMergeTypes(F1, F2, Options)) {
    LLVM_DEBUG(dbgs() << "Simple return types\n");
    ReturnType = RetType1;
    if (ReturnType->isVoidTy()) {
      ReturnType = RetType2;
    }
  } else if (Options.EnableUnifiedReturnType) {
    LLVM_DEBUG(dbgs() << "Unifying return types\n");
    RequiresUnifiedReturn = true;

    const DataLayout *DL = &F1->getParent()->getDataLayout();
    auto SizeOfTy1 = DL->getTypeStoreSize(RetType1);
    auto SizeOfTy2 = DL->getTypeStoreSize(RetType2);
    if (SizeOfTy1 >= SizeOfTy2) {
      ReturnType = RetType1;
    } else {
      ReturnType = RetType2;
    }
  }

  return ReturnType!=nullptr;
}

FunctionMergeResult CodeMerger::defineMergedFunction(Function *F1, Function *F2, const char *Name, CodeAlignment *AR, ValueToValueMapTy &VMap, const FunctionMergingOptions &Options) {
  // Merging parameters
  std::map<unsigned, unsigned> ParamMap1;
  std::map<unsigned, unsigned> ParamMap2;
  std::vector<Type *> Args;

  MergeArguments(getContext(), F1, F2, AR, ParamMap1,ParamMap2,Args,Options);
  
  defineReturnType(F1, F2, Options);

  FunctionType *FTy = FunctionType::get(getReturnType(), ArrayRef<Type*>(Args), false);

  F = Function::Create(FTy, GlobalValue::LinkageTypes::InternalLinkage, Name, M);

  FunctionMergeResult Result(F1,F2,F,RequiresUnifiedReturn);
  Result.setArgumentMapping(F1,ParamMap1);
  Result.setArgumentMapping(F2,ParamMap2);
  Result.setFunctionIdArgument(true);

  std::vector<Argument *> ArgsList;
  for (Argument &arg : F->args()) {
    ArgsList.push_back(&arg);
  }
  Value *FuncId = ArgsList[0];

  int ArgId = 0;
  for (auto I = F1->arg_begin(), E = F1->arg_end(); I != E; I++) {
    VMap[&(*I)] = ArgsList[ParamMap1[ArgId]];
    ArgId++;
  }

  ArgId = 0;
  for (auto I = F2->arg_begin(), E = F2->arg_end(); I != E; I++) {
    VMap[&(*I)] = ArgsList[ParamMap2[ArgId]];
    ArgId++;
  }

  SetFunctionAttributes(F1,F2,F);

  IsFirst = FuncId;

  setEntryPoint(BasicBlock::Create(getContext(), "entry", F));
  setReturnType(ReturnType, RequiresUnifiedReturn);

  return Result;
}

bool CodeMerger::generate(CodeAlignment *AR,
                  ValueToValueMapTy &VMap,
                  const FunctionMergingOptions &Options) {

  LLVMContext &Context = getContext();
  Function *MergedFunc = getFunction();
  Value *IsFirst = getCondition();
  BasicBlock *PreBB = getPreBlock();

  auto Blocks1 = getBlocks1();
  auto Blocks2 = getBlocks2();

  std::list<Instruction *> LinearOffendingInsts;
  std::set<Instruction *> OffendingInsts;

  //maps new basic blocks in the merged function to their original correspondents
  std::unordered_map<BasicBlock *, BasicBlock *> BlocksF1;
  std::unordered_map<BasicBlock *, BasicBlock *> BlocksF2;
  std::list<MergedInstruction> MergedInsts;

  codeGen(AR,VMap,BlocksF1,BlocksF2,MergedInsts);

  std::set<BranchInst*> XorBrConds;
  //assigning label operands
  
  for (auto &MI : MergedInsts) {
    Instruction *I1 = MI[0];
    Instruction *I2 = MI[1];

    if (I1!=nullptr && I2!=nullptr) {

      Instruction *I = I1;
      if (I1->getOpcode() == Instruction::Ret) {
        I = (I1->getNumOperands() >= I2->getNumOperands())? I1 : I2 ;
      } else {
        assert(I1->getNumOperands() == I2->getNumOperands() &&
               "Num of Operands SHOULD be EQUAL\n");
      }

      Instruction *NewI = MI.get();

      bool Handled = false;
      
      BranchInst *NewBr = dyn_cast<BranchInst>(NewI);    
      if (EnableOperandReordering && NewBr!=nullptr && NewBr->isConditional()) { 
         BranchInst *Br1 = dyn_cast<BranchInst>(I1);       
         BranchInst *Br2 = dyn_cast<BranchInst>(I2);
         
         BasicBlock *SuccBB10 = dyn_cast<BasicBlock>(MapValue(Br1->getSuccessor(0), VMap));
         BasicBlock *SuccBB11 = dyn_cast<BasicBlock>(MapValue(Br1->getSuccessor(1), VMap));

         BasicBlock *SuccBB20 = dyn_cast<BasicBlock>(MapValue(Br2->getSuccessor(0), VMap));
         BasicBlock *SuccBB21 = dyn_cast<BasicBlock>(MapValue(Br2->getSuccessor(1), VMap));

         if (SuccBB10!=nullptr && SuccBB11!=nullptr && SuccBB10==SuccBB21 && SuccBB20==SuccBB11) {
             LLVM_DEBUG(dbgs() << "OptimizationTriggered: Labels of Conditional Branch Reordering\n");

             XorBrConds.insert(NewBr);
             NewBr->setSuccessor(0,SuccBB20);
             NewBr->setSuccessor(1,SuccBB21);
             Handled = true;
         }
      }

      if (!Handled) {
        for (unsigned i = 0; i < I->getNumOperands(); i++) {

          Value *F1V = nullptr;
          Value *V1 = nullptr;
          if (i < I1->getNumOperands()) {
            F1V = I1->getOperand(i);
	    if (!isa<BasicBlock>(F1V))
	      continue;
            V1 = MapValue(F1V, VMap);
          } else {
            V1 = UndefValue::get(I2->getOperand(i)->getType());
          }

          Value *F2V = nullptr;
          Value *V2 = nullptr;
          if (i < I2->getNumOperands()) {
            F2V = I2->getOperand(i);
	    if (!isa<BasicBlock>(F2V))
	      continue;
            V2 = MapValue(F2V, VMap);
          } else {
            V2 = UndefValue::get(I1->getOperand(i)->getType());
          }

          assert(V1 != nullptr && "Value should NOT be null!");
          assert(V2 != nullptr && "Value should NOT be null!");

          Value *V = V1; // first assume that V1==V2

          //handling just label operands for now
          if (!isa<BasicBlock>(V))
            continue;

          BasicBlock *F1BB = dyn_cast<BasicBlock>(F1V);
          BasicBlock *F2BB = dyn_cast<BasicBlock>(F2V);

          if (V1 != V2) {
            BasicBlock *BB1 = dyn_cast<BasicBlock>(V1);
            BasicBlock *BB2 = dyn_cast<BasicBlock>(V2);

            BasicBlock *SelectBB = BasicBlock::Create(Context, "bb.select", MergedFunc);
            IRBuilder<> BuilderBB(SelectBB);

            BlocksF1[SelectBB] = I1->getParent();
            BlocksF2[SelectBB] = I2->getParent();

            BuilderBB.CreateCondBr(IsFirst, BB1, BB2);
            V = SelectBB;
          }
          
          if (F1BB->isLandingPad() || F2BB->isLandingPad()) {
            LandingPadInst *LP1 = F1BB->getLandingPadInst();
            LandingPadInst *LP2 = F2BB->getLandingPadInst();
            assert((LP1 != nullptr && LP2 != nullptr) &&
                   "Should be both as per the BasicBlock match!");

            BasicBlock *LPadBB = BasicBlock::Create(Context, "lpad.bb", MergedFunc);
            IRBuilder<> BuilderBB(LPadBB);

            Instruction *NewLP = LP1->clone();
            BuilderBB.Insert(NewLP);

            BuilderBB.CreateBr(dyn_cast<BasicBlock>(V));
 
            BlocksF1[LPadBB] = I1->getParent();
            BlocksF2[LPadBB] = I2->getParent();

            VMap[LP1] = NewLP;
            VMap[LP2] = NewLP; 
            
            V = LPadBB;         
          }
          NewI->setOperand(i, V);
        }
      }

    } else {
      if (I1)
        assignLabelOperands(I1, BlocksF1, VMap);
      if (I2)
        assignLabelOperands(I2, BlocksF2, VMap);
    }

  }

  for (auto &MI : MergedInsts) {
    Instruction *I1 = MI[0];
    Instruction *I2 = MI[1];

    // Skip non-instructions
    if (I1==nullptr && I2==nullptr) {
      errs() << "ERROR: NULL Instructions\n";
      continue;
    }
 
    if ( isa<PHINode>(MI.get()) ) continue;

    if (I1!=nullptr && I2!=nullptr) {

      Instruction *I = I1;
      if (I1->getOpcode() == Instruction::Ret) {
        I = (I1->getNumOperands() >= I2->getNumOperands()) ? I1 : I2;
      } else {
        assert(I1->getNumOperands() == I2->getNumOperands() &&
               "Num of Operands SHOULD be EQUAL\n");
      }

      Instruction *NewI = MI.get();

      IRBuilder<> Builder(NewI);

      if (EnableOperandReordering && isa<BinaryOperator>(NewI) && NewI->isCommutative()) {

        BinaryOperator *BO1 = dyn_cast<BinaryOperator>(I1);
        BinaryOperator *BO2 = dyn_cast<BinaryOperator>(I2);
        Value *VL1 = MapValue(BO1->getOperand(0), VMap);
        Value *VL2 = MapValue(BO2->getOperand(0), VMap);
        Value *VR1 = MapValue(BO1->getOperand(1), VMap);
        Value *VR2 = MapValue(BO2->getOperand(1), VMap);
        if (VL1 == VR2 && VL2 != VR2) {
          std::swap(VL2, VR2);
        } else if (VL2 == VR1 && VL1 != VR1) {
          std::swap(VL1, VR1);
        }

        std::vector<std::pair<Value *, Value *>> Vs;
        Vs.push_back(std::pair<Value *, Value *>(VL1, VL2));
        Vs.push_back(std::pair<Value *, Value *>(VR1, VR2));

        for (unsigned i = 0; i < Vs.size(); i++) {
          Value *V1 = Vs[i].first;
          Value *V2 = Vs[i].second;
          Value *V = mergeValues(V1, V2, NewI);
          NewI->setOperand(i, V);
        }
      } else {
        for (unsigned i = 0; i < NewI->getNumOperands(); i++) {
          if (isa<BasicBlock>(I->getOperand(i)))
            continue;

          Value *V1 = nullptr;
          if (i < I1->getNumOperands()) {
            V1 = MapValue(I1->getOperand(i), VMap);
            assert(V1!=nullptr && "Mapped value should NOT be NULL!");
          } else {
            V1 = UndefValue::get(I2->getOperand(i)->getType());
          }

          Value *V2 = nullptr;
          if (i < I2->getNumOperands()) {
            V2 = MapValue(I2->getOperand(i), VMap);
            assert(V2!=nullptr && "Mapped value should NOT be NULL!");
          } else {
            V2 = UndefValue::get(I1->getOperand(i)->getType());
          }

          assert(V1 != nullptr && "Value should NOT be null!");
          assert(V2 != nullptr && "Value should NOT be null!");

          Value *V = mergeValues(V1, V2, NewI);
          NewI->setOperand(i, V);

        } // end for operands
      }
    } // end if isomorphic
    else {
      if (I1)
        assignOperands(I1, true, VMap);
      
      if (I2)
        assignOperands(I2, false, VMap);
    } // end 'if-else' non-isomorphic

  } // end for nodes

  if (AllSelections.size() > MaxNumSelection) {
    LLVM_DEBUG(dbgs() << "Bailing out: Operand selection threshold\n");
    return false;
  }

  for (BasicBlock *BB1 : Blocks1) {
      if (!assignPHIOperandsInBlock(BB1, BlocksF1, VMap)) {
          errs() << "ERROR: PHI assignment\n";
          return false;
      }
  }
  for (BasicBlock *BB2 : Blocks2) {
      if (!assignPHIOperandsInBlock(BB2, BlocksF2, VMap)) {
          errs() << "ERROR: PHI assignment\n";
          return false;
      }
  }

  for (BasicBlock &BB : *MergedFunc) {
    std::vector<PHINode*> AllPHIs;
    auto It = BB.begin();
    auto *EndI = BB.getFirstNonPHI();
    while (It!=BB.end() && (&*It)!=EndI) {
      Instruction *I = &*It;
      It++;
      if (PHINode *PHI = dyn_cast<PHINode>(I)) {
	PHINode *EquivPHI = nullptr;
	for (PHINode *OtherPHI : AllPHIs) {
          bool IsEqual = true;
          for (unsigned i = 0; i<PHI->getNumIncomingValues(); i++) {
            IsEqual = IsEqual && OtherPHI->getIncomingValueForBlock(PHI->getIncomingBlock(i))==PHI->getIncomingValue(i);
	  }
	  if (IsEqual) {
	    EquivPHI = OtherPHI;
            break;
	  }
	}
	if (EquivPHI) {
	  PHI->replaceAllUsesWith(EquivPHI);
	  PHI->eraseFromParent();
	} else AllPHIs.push_back(PHI);
      }
    }
  }

  for (auto *SelI : AllSelections) {
    if (SelI->getTrueValue()==SelI->getFalseValue()) {
      SelI->replaceAllUsesWith(SelI->getTrueValue());
      SelI->eraseFromParent();
    }
  }
  AllSelections.clear();

  DominatorTree DT(*MergedFunc);

  for (Instruction &I : instructions(MergedFunc)) {
    if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
      for (unsigned i = 0; i<PHI->getNumIncomingValues(); i++) {
        BasicBlock *BB = PHI->getIncomingBlock(i);
	if (BB==nullptr) errs() << "ERROR: Null incoming block\n";
	Value *V = PHI->getIncomingValue(i);
	if (V==nullptr) errs() << "ERROR: Null incoming value\n";
        if (Instruction *IV = dyn_cast<Instruction>(V)) {
	  if (BB->getTerminator()==nullptr) errs() << "ERROR: Null terminator\n";
          if (!DT.dominates(IV,BB->getTerminator())) {
            if (OffendingInsts.count(IV)==0) { OffendingInsts.insert(IV); LinearOffendingInsts.push_back(IV); }
	  }
	}
      }
    } else {
      for (unsigned i = 0; i<I.getNumOperands(); i++) {
	if (I.getOperand(i)==nullptr) {
		I.getParent()->dump();
		errs() << "ERROR: Null operand\n";
		I.dump();
	}
        if (Instruction *IV = dyn_cast<Instruction>(I.getOperand(i))) {
	  if (!DT.dominates(IV, &I)) {
            if (OffendingInsts.count(IV)==0) { OffendingInsts.insert(IV); LinearOffendingInsts.push_back(IV); }
          }
        }
      }
    }
  }


  for (BranchInst *NewBr : XorBrConds) {
    IRBuilder<> Builder(NewBr);
    Value *XorCond = Builder.CreateXor(NewBr->getCondition(),IsFirst);
    NewBr->setCondition(XorCond);
  }

  if (MergedFunc!=nullptr) {
    if (OffendingInsts.size()>1000) {
      LLVM_DEBUG(dbgs() << "Bailing out: offending instructions\n");
      return false;
    } else {
      std::vector<AllocaInst *> Allocas;
      for (Instruction *I : LinearOffendingInsts) {
        AllocaInst *Addr = memfyInst(I);
        Allocas.push_back( Addr );
      }
      DominatorTree DT(*MergedFunc);
      PromoteMemToReg(Allocas, DT, nullptr);

      if (PreBB->getSingleSuccessor()) {
        MergeBlockIntoPredecessor(PreBB->getSingleSuccessor());
      }
    }
  }

  return MergedFunc!=nullptr;
}

