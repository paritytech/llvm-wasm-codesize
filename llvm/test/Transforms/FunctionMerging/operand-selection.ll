; RUN: opt < %s -func-merging -func-merging-force -S | FileCheck %s

; Function Attrs: minsize nofree noinline norecurse nosync nounwind optsize readnone uwtable willreturn mustprogress
define dso_local i32 @f1(i32 %c, i32 %d) local_unnamed_addr {
; CHECK:    %1 = tail call i32 @.m.f.0(i1 true, i32 %c, i32 %d)
; CHECK-NEXT:    ret i32 %1
entry:
  %add = add nsw i32 %d, %c
  %mul = shl nsw i32 %add, 1
  ret i32 %mul
}

; Function Attrs: minsize nofree noinline norecurse nosync nounwind optsize readnone uwtable willreturn mustprogress
define dso_local i32 @f2(i32 %c, i32 %d) local_unnamed_addr {
; CHECK:    %1 = tail call i32 @.m.f.0(i1 false, i32 %c, i32 %d)
; CHECK-NEXT:    ret i32 %1
entry:
  %add = add nsw i32 %d, %c
  %mul = shl nsw i32 %add, 2
  ret i32 %mul
}

; CHECK-LABEL: define private i32 @.m.f.0(i1 %0, i32 %1, i32 %2) local_unnamed_addr
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %3 = add nsw i32 %2, %1
; CHECK-NEXT:    %4 = select i1 %0, i32 1, i32 2
; CHECK-NEXT:    %5 = shl nsw i32 %3, %4
; CHECK-NEXT:    ret i32 %5

