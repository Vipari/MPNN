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
	learnRate  float64    // Scales how quickly SGD should work [Too small = Learns slow -- Too big = Doesn't minimize cost function]
}

func initRandArray(size int, fromSize float64) []float64 {
	var arr = make([]float64, size)

	// Sets a uniform range between +-1 / sqrt(size of last layer), ensures network starts off with unsure predictions.
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(fromSize),
		Max: 1 / math.Sqrt(fromSize),
		Src: rand.NewSource(uint64(time.Now().UnixNano())),
	}

	// Unscaled random
	// dist := distuv.Uniform{
	// 	Min: -1,
	// 	Max: 1,
	// 	Src: rand.NewSource(uint64(time.Now().UnixNano())),
	// }

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

	// Create weight matrix in between each neuron layer.
	// # of Inputs = # of Columns
	// # of Outputs = # of Rows
	// Simplifies the math to a few matrix operations this way.

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

// This is where the network updates the weights based on gradient descent. (Training)
func (net *MPNN) backProp(input []float64, target []float64) {

	// Forward Propagation
	// Can't use fowardProp() because intermediary values are needed
	inLayer := mat.NewDense(len(input), 1, input)

	inLayerWeightsIn := dot(net.hidWeights, inLayer)
	inLayerWeightsOut := apply(sigmoid, inLayerWeightsIn)

	hidLayerWeightsIn := dot(net.outWeights, inLayerWeightsOut)
	hidLayerWeightsOut := apply(sigmoid, hidLayerWeightsIn)

	// Find error
	// Difference between predicted output and actual value
	actual := mat.NewDense(len(target), 1, target)      // Target data
	outputError := sub(actual, hidLayerWeightsOut)      // How far the predicted output is from the target data
	hiddenError := dot(net.outWeights.T(), outputError) // Calculus to find hidden layer error from the output error

	// Back Propagation
	// Adjust each weight a little bit by the error of the next layer, going from the output back towards the input.

	// Adjust the output layer weights [hidden -> output] by the output error
	//This neat little bit of calculus calculates the needed change in weights and adjusts the weights using that.
	net.outWeights = add(net.outWeights,
		scale(net.learnRate,
			dot(mult(outputError, sigmoidDerivative(hidLayerWeightsOut)),
				inLayerWeightsOut.T()))).(*mat.Dense)

	// Adjust hidden layer weights [input -> hidden] by the hidden error
	net.hidWeights = add(net.hidWeights,
		scale(net.learnRate,
			dot(mult(hiddenError, sigmoidDerivative(inLayerWeightsOut)),
				inLayer.T()))).(*mat.Dense)

	// ***Haven't gotten to it yet, but all you would have to do now is load it up with some training data and save the weight's
	// values for future use (so you don't have to train every time you run the program)!

}

// Since matricies and vectors are interfaces and not types, functions on them don't return values,
// which can make them unwieldy to deal with when doing many operations on them, so it's common to
// create helper functions to do these operations in a more traditonal manor.

// p# are placeholders so I can use the function on Matrix.Apply().
func sigmoid(p1, p2 int, x float64) float64 { // Squishes input between 0 and 1, resembles a smooth step function.
	return 1 / (1 + math.Exp(-x))
}
func sigmoidDerivative(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return mult(m, sub(ones, m))
}

func dot(m mat.Matrix, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	out := mat.NewDense(r, c, nil)
	out.Product(m, n)
	return out
}
func scale(factor float64, m mat.Matrix) mat.Matrix {
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

func printMatrix(m mat.Matrix) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if m.At(i, j) > 0 {
				fmt.Print(" ")
			}
			fmt.Printf("%.4f ", m.At(i, j))
		}
		fmt.Println()
	}
	fmt.Println()
}

func main() {
	var net MPNN = initMPNN([]int{10, 20, 5}, 0.01)

	randInput := initRandArray(net.in, 1)
	guess := forwardProp(randInput, net)

	fmt.Println("[Input Layer -> Hidden Layer Matrix]")
	printMatrix(net.hidWeights)

	fmt.Println("[Hidden Layer-> Output Layer Matrix]")
	printMatrix(net.outWeights)

	fmt.Println("[Guess Matrix]")
	printMatrix(guess)
}
