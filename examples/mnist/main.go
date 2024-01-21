package main

import (
	"fmt"
	"log"
	"time"

	"github.com/tsukanov-as/watf"
)

func main() {
	train, err := readMnistCsv("mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}
	test, err := readMnistCsv("mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	w := watf.New(10, 28*28, watf.WithPenalization())

	start := time.Now()
	// tune
	for epoch := 0; epoch < 50; epoch++ {
		for _, r := range train {
			w.Tune(r[0], r[1:])
		}

		total := 0.0
		for _, r := range test {
			if w.Predict(r[1:]) == r[0] {
				total += 1
			}
		}
		fmt.Printf("accuracy on test: %f\n", total/float64(len(test)))
	}
	fmt.Println("time:", time.Since(start))
}
