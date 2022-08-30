package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println(mat.NewVecDense(3, []float64{1, 2, 3}).RawVector().Data)
}
