package main

import (
	"fmt"
	"log"
	"time"

	"github.com/tsukanov-as/watf"
	"github.com/tsukanov-as/watf/mix"
)

type Number = int
type Watf = watf.Watf[Number]

func main() {
	train, err := readMnistCsv("mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}
	// train = train[:10000]
	test, err := readMnistCsv("mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}
	// test = test[:1000]

	levels := 6

	m := mix.New[Number](10, 28*28, levels)

	start := time.Now()

	for _, r := range train {
		m.FeedRandom(r[1:])
	}
	m.Commit()

	for level := 0; level < levels; level++ {
		for epoch := 0; epoch < 10; epoch++ {
			for _, r := range train {
				m.Feed(r[1:], m.Cluster(r[1:], level))
			}
			m.Commit()
			fmt.Printf("sharding level: %d, epoch: %d\n", level, epoch)
		}
	}

	shards_train := make([]*Watf, len(train))
	for i, r := range train {
		shards_train[i] = m.Shard(r[1:])
	}

	shards_test := make([]*Watf, len(test))
	for i, r := range test {
		shards_test[i] = m.Shard(r[1:])
	}

	// tune
	for epoch := 0; epoch < 100; epoch++ {
		for i, r := range train {
			shards_train[i].Tune(r[0], r[1:])
		}

		total_train := 0.0
		for i, r := range train {
			if shards_train[i].Predict(r[1:]) == r[0] {
				total_train += 1
			}
		}

		total_test := 0.0
		for i, r := range test {
			if shards_test[i].Predict(r[1:]) == r[0] {
				total_test += 1
			}
		}
		fmt.Printf("[epoch %d] accuracy train: %f; accuracy test: %f\n", epoch, total_train/float64(len(train)), total_test/float64(len(test)))
	}
	fmt.Println("time:", time.Since(start))
}
