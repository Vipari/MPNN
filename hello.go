package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

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
	dist := distuv.Uniform{Min: 1 / math.Sqrt(fromSize), Max: -1 / math.Sqrt(fromSize)} // Sets a uniform range between +-1 / sqrt(size of last layer)
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
		network.in, network.hidden,
		initRandArray(network.hidden*network.in, float64(network.in)))
	network.outWeights = mat.NewDense(
		network.hidden, network.out,
		initRandArray(network.hidden*network.out, float64(network.hidden)))

	return network
}

func printMPNN(network MPNN) { // Prints the weights of the network

	fmt.Println("{Key: i2.h4 reads as \"Input Neuron 2 to Hidden Neuron 4\"}")

	fmt.Println("\nLearn Rate: ", network.learnRate)

	fmt.Println("[Input -> Hidden]")

	for i := 0; i < network.in; i++ {
		fmt.Print("Input ", i, ": ")

		for j := 0; j < network.hidden; j++ {

			fmt.Print(" i", i, ".h", j, ":")
			if network.hidWeights.At(i, j) > 0 {
				fmt.Print(" ")
			}

			fmt.Printf("%.4f  ", network.hidWeights.At(i, j))
		}
		fmt.Println()
	}

	fmt.Println("\n[Hidden -> Output]")

	for i := 0; i < network.hidden; i++ {
		fmt.Print("Hidden ", i, ": ")

		for j := 0; j < network.out; j++ {

			fmt.Print(" h", i, ".o", j, ":")
			if network.outWeights.At(i, j) > 0 {
				fmt.Print(" ")
			}

			fmt.Printf("%.4f  ", network.outWeights.At(i, j))
		}
		fmt.Println()
	}
}

func main() {
	var net MPNN = initMPNN([]int{5, 10, 5}, 0.01)
	printMPNN(net)
}
