package main

import (
	"fmt"
	"math"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// A slightly more advanced network might take advantage of biases (x-axis translations of the sigmoid function),
// but our simple network can achieve decent performance without them, so they have been omitted from the network.

type MPNN struct { // 3 Layer Neural Network
	in         int
	hidden     int
	out        int
	hidWeights *mat.Dense // Matrix for input layer -> hidden layer weights
	outWeights *mat.Dense // Matrix for hidden layer -> input layer weights
	learnRate  float64    // Scales how quickly SGD should work [Too small = Learns slow --- Too big = Doesn't minimize cost function]
}

func initRandArray(size int, fromSize float64) []float64 {
	var arr = make([]float64, size)
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(fromSize),
		Max: 1 / math.Sqrt(fromSize),
		Src: rand.NewSource(uint64(time.Now().UnixNano())),
	} // Sets a uniform range between +-1 / sqrt(size of last layer)

	// dist := distuv.Uniform{
	// 	Min: -1,
	// 	Max: 1,
	// 	Src: rand.NewSource(uint64(time.Now().UnixNano())),
	// } // Unscaled random

	for i := range arr {
		arr[i] = dist.Rand()
	}
	return arr
}

func initMPNN(sizes []int, learn float64) (network MPNN) {

	network = MPNN{
		in:        sizes[0],
		hidden:    sizes[1],
		out:       sizes[2],
		learnRate: learn,
	}

	//Create matrixes for weights of neuron all neuron pairs between layers

	network.hidWeights = mat.NewDense(
		network.hidden, network.in,
		initRandArray(network.hidden*network.in, float64(network.in)))
	network.outWeights = mat.NewDense(
		network.out, network.hidden,
		initRandArray(network.hidden*network.out, float64(network.hidden)))

	return network
}

// This is where the network "predicts" and we get our output.
// Forward propagation is the algorithm that takes in the input, and calculates the output of each
// consecutive layer using the weights until reaching the output layer.
// σ(W ⋅ A)
func forwardProp(input []float64, network MPNN) mat.Matrix {
	inLayer := mat.NewDense(len(input), 1, input)
	inLayerWeightsIn := dot(network.hidWeights, inLayer)
	inLayerWeightsOut := apply(sigmoid, inLayerWeightsIn)
	hidLayerWeightsIn := dot(network.outWeights, inLayerWeightsOut)
	hidLayerWeightsOut := apply(sigmoid, hidLayerWeightsIn)
	return hidLayerWeightsOut

}

// This is where the network updates the weights bases on stochastic gradient descent. (Training)
func backProp() {

}

// Since matricies and vectors are interfaces and not types, functions on them don't return values,
// which can make them unwieldy to deal with when doing many operations on them, so it's common to
// create helper functions to do these operations in a more traditonal manor.

// p# are placeholders so I can use the function on Matrix.Apply().
func sigmoid(p1, p2 int, x float64) float64 { // Squishes input between 0 and 1, resembles a smooth step function.
	return 1 / (1 + math.Exp(-x))
}
func sigmoidDerivative(x float64) float64 {
	return sigmoid(0, 0, x) * (1 - sigmoid(0, 0, x))
}

func dot(m mat.Matrix, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	out := mat.NewDense(r, c, nil)
	out.Product(m, n)
	return out
}
func scale(m mat.Matrix, factor float64) mat.Matrix {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	out.Scale(factor, m)
	return out
}
func mult(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	out.MulElem(m, n)
	return out
}
func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	out.Add(m, n)
	return out
}
func sub(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	out.Sub(m, n)
	return out
}
func scalar(m mat.Matrix, scalar float64) mat.Matrix {
	r, c := m.Dims()
	s := make([]float64, r*c)
	for i, _ := range s {
		s[i] = scalar
	}
	n := mat.NewDense(r, c, s)
	return add(m, n)
}
func apply(fn func(i, j int, f float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	out.Apply(fn, m)
	return out
}

func printMPNN(network MPNN) { // Prints the weights of the network

	fmt.Println("{Key: i2.h4 reads as \"Input Neuron 2 to Hidden Neuron 4\"}")

	fmt.Println("\nLearn Rate: ", network.learnRate)

	fmt.Println("[Input -> Hidden]")

	for i := 0; i < network.hidden; i++ {
		fmt.Print("Input ", i, ": ")

		for j := 0; j < network.in; j++ {

			fmt.Print(" i", i, ".h", j, ":")
			if network.hidWeights.At(i, j) > 0 {
				fmt.Print(" ")
			}

			fmt.Printf("%.4f  ", network.hidWeights.At(i, j))
		}
		fmt.Println()
	}

	fmt.Println("\n[Hidden -> Output]")

	for i := 0; i < network.out; i++ {
		fmt.Print("Hidden ", i, ": ")

		for j := 0; j < network.hidden; j++ {

			fmt.Print(" h", i, ".o", j, ":")
			if network.outWeights.At(i, j) > 0 {
				fmt.Print(" ")
			}

			fmt.Printf("%.4f  ", network.outWeights.At(i, j))
		}
		fmt.Println()
	}
}
func printMatrix(m mat.Matrix) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Print("Output Neuron ", i, ": ", m.At(i, j), " ")
		}
		fmt.Println()
	}
}

func main() {
	var net MPNN = initMPNN([]int{5, 10, 5}, 0.01)
	guess := forwardProp([]float64{
		0.25,
		0.1,
		0.99,
		0.62,
		0.01,
	}, net)
	printMPNN(net)
	printMatrix(guess)
}
