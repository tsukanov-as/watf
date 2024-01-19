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

func (w *Watf) Feed(cl int, p int, fv []int) {
	base := cl * w.fc
	baseP := p * w.fc
	for i := 0; i < w.fc; i++ {
		w.wv[base+i] += fv[i]
		w.wv[baseP+i] -= fv[i] / 2
	}
}

func (w *Watf) Pred(fv []int) int {
	// argmax(weights @ features)
	maxv := 0
	maxc := 0
	base := 0
	for cl := 0; cl < w.cc; cl++ {
		total := 0
		wv := w.wv[base : base+w.fc]
		for i, v := range wv {
			total += fv[i] * v
		}
		if maxv < total {
			maxv = total
			maxc = cl
		}
		base += w.fc
	}
	return maxc
}

func (watf *Watf) Tune(cl int, fv []int) bool {
	p := watf.Pred(fv)
	if p != cl {
		watf.Feed(cl, p, fv)
		return true
	}
	return false
}
