package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
)

const (
	mnistImgLen = 28 * 28
	mnistRecLen = 1 + mnistImgLen
)

func readMnistCsv(path string) ([][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal("Unable to read input file "+path, err)
	}
	defer f.Close()

	var mnist [][]int

	rd := csv.NewReader(f)
	for {
		r, err := rd.Read()
		if r == nil || err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(r) != mnistRecLen {
			return nil, fmt.Errorf("unxepected record len: %d", len(r))
		}
		vec := make([]int, mnistRecLen)
		mnist = append(mnist, vec)
		cl, err := strconv.Atoi(r[0])
		if err != nil {
			return nil, err
		}
		vec[0] = cl
		for i := 1; i < mnistRecLen; i++ {
			if r[i] != "0" {
				v, err := strconv.Atoi(r[i])
				if err != nil {
					return nil, err
				}
				vec[i] = v
			}
		}
	}
	return mnist, nil
}
