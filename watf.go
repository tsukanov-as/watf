package watf

type number interface {
	~int | ~int32 | ~int64 | ~float32 | ~float64
}

type options struct {
	penalize bool
}

type Watf[T number] struct {
	classes  int
	features int
	weights  []T
	results  []T
	options
}

type option func(watf *options)

func WithPenalization() option {
	return func(watf *options) {
		watf.penalize = true
	}
}

func New[T number](classes, features int, options ...option) *Watf[T] {
	watf := &Watf[T]{
		classes:  classes,
		features: features,
		weights:  make([]T, classes*features),
		results:  make([]T, classes),
	}
	for _, opt := range options {
		opt(&watf.options)
	}
	return watf
}

func (watf *Watf[T]) Feed(y int, x []T) {
	w := watf.weights[y*watf.features:]
	for i := 0; i < watf.features; i++ {
		w[i] += x[i]
	}
}

func (watf *Watf[T]) Penalize(y int, x []T) {
	w := watf.weights[y*watf.features:]
	for i := 0; i < watf.features; i++ {
		w[i] -= x[i] / 2
	}
}

func (watf *Watf[T]) Predict(x []T) int {
	// argmax(weights @ features)
	base := 0
	for y := 0; y < watf.classes; y++ {
		w := watf.weights[base : base+watf.features]
		watf.results[y] = dot(w, x)
		base += watf.features
	}
	return argmax(watf.results)
}

func (watf *Watf[T]) Tune(y int, x []T) bool {
	p := watf.Predict(x)
	if p != y {
		watf.Feed(y, x)
		if watf.penalize {
			watf.Penalize(p, x)
		}
		return true
	}
	return false
}

func argmax[T number](a []T) int {
	mi := 0
	mv := a[0]
	for i, v := range a {
		if v > mv {
			mi = i
			mv = v
		}
	}
	return mi
}

func dot[T number](a, b []T) T {
	var sum T
	for i, v := range a {
		sum += b[i] * v
	}
	return sum
}
