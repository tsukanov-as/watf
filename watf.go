package watf

type Watf struct {
	cc int
	fc int
	wv []int
}

func New(classes, features int) *Watf {
	return &Watf{
		cc: classes,
		fc: features,
		wv: make([]int, classes*features),
	}
}

func (w *Watf) Feed(cl int, fv []int) {
	base := cl * w.fc
	for i := 0; i < w.fc; i++ {
		w.wv[base+i] += fv[i]
	}
}

func (w *Watf) Pred(fv []int) int {
	// argmax(weights @ features)
	maxv := 0
	maxc := 0
	for cl := 0; cl < w.cc; cl++ {
		total := 0
		base := cl * w.fc
		wv := w.wv[base : base+w.fc]
		for i, v := range wv {
			total += fv[i] * v
		}
		if maxv < total {
			maxv = total
			maxc = cl
		}
	}
	return maxc
}

func (watf *Watf) Tune(cl int, fv []int) bool {
	if watf.Pred(fv) != cl {
		watf.Feed(cl, fv)
		return true
	}
	return false
}
