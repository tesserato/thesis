var A = []

A.push(
  {
    l1:"label 01",
    l2:"label 02"
  }
)

A.forEach(e => {e.l1 = "new label 01"
  
});



console.log(A[0].l1)