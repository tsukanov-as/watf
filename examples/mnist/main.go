package main

import (
	"fmt"
	"log"

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

	w := watf.New(10, 28*28)

	// train
	for _, r := range train {
		w.Feed(r[0], r[1:])
	}

	// test
	total := 0.0
	for _, r := range test {
		if w.Pred(r[1:]) == r[0] {
			total += 1
		}
	}
	fmt.Println(total / float64(len(test)))

	// tune
	for epoch := 0; epoch < 20; epoch++ {
		for _, r := range train {
			w.Tune(r[0], r[1:])
		}

		total := 0.0
		for _, r := range test {
			if w.Pred(r[1:]) == r[0] {
				total += 1
			}
		}
		fmt.Println(total / float64(len(test)))
	}
}
