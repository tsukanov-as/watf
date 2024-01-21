package watf

type Watf struct {
	classes  int
	features int
	weights  []int
	penalize bool
}

type option func(watf *Watf)

func WithPenalization() option {
	return func(watf *Watf) {
		watf.penalize = true
	}
}

func New(classes, features int, options ...option) *Watf {
	watf := &Watf{
		classes:  classes,
		features: features,
		weights:  make([]int, classes*features),
	}
	for _, opt := range options {
		opt(watf)
	}
	return watf
}

func (watf *Watf) Feed(y int, x []int) {
	base := y * watf.features
	for i := 0; i < watf.features; i++ {
		watf.weights[base+i] += x[i]
	}
}

func (watf *Watf) Penalize(y int, x []int) {
	base := y * watf.features
	for i := 0; i < watf.features; i++ {
		watf.weights[base+i] -= x[i] / 2
	}
}

func (watf *Watf) Predict(x []int) int {
	// argmax(weights @ features)
	vMax := 0
	yMax := 0
	base := 0
	for y := 0; y < watf.classes; y++ {
		w := watf.weights[base : base+watf.features]
		if v := dot(w, x); vMax < v {
			vMax = v
			yMax = y
		}
		base += watf.features
	}
	return yMax
}

func (watf *Watf) Tune(y int, x []int) bool {
	p := watf.Predict(x)
	if p != y {
		watf.Feed(y, x)
		if watf.penalize {
			watf.Penalize(p, x)
			return true
		}
		return true
	}
	return false
}

func dot(a, b []int) int {
	sum := 0
	for i, v := range a {
		sum += b[i] * v
	}
	return sum
}
