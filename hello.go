package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	u := mat.NewVecDense(3, []float64{1, 2, 3})
	v := mat.NewVecDense(3, []float64{4, 5, 6})

	fmt.Println(u.AtVec(1))
	fmt.Println(v.AtVec(2))

	u.AddVec(u, v)
	fmt.Println(u.RawVector().Data)
}
