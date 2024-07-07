extern crate ndarray;
use ndarray::*;
use std::rc::Rc;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::ops::*;

#[derive(Debug, Clone)]
struct Value {
    data: f64,
    _op: char,
    _prev: HashSet<Rc<Value>>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            _prev: HashSet::new(),
            _op:' '
        }
    }

    pub fn init(&self) {
        println!("{}", self.data)
    }
}

impl Add for Value {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut new_prev = self._prev.clone();
        new_prev.insert(Rc::new(self.clone()));
        new_prev.insert(Rc::new(other.clone()));
        Self {
            data: self.data + other.data,
            _prev: new_prev,
            _op:'+'
        }
    }
}

impl Sub for Value {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let mut new_prev = self._prev.clone();
        new_prev.insert(Rc::new(self.clone()));
        new_prev.insert(Rc::new(other.clone()));
        Self {
            data: self.data - other.data,
            _prev: new_prev,
            _op:'-'
        }
    }
}

impl Mul for Value {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let mut new_prev = self._prev.clone();
        new_prev.insert(Rc::new(self.clone()));
        new_prev.insert(Rc::new(other.clone()));
        Self {
            data: self.data * other.data,
            _prev: new_prev,
            _op:'*'
        }
    }
}

fn main() {
    let mut arr = Array2::from_shape_vec((3, 3), (1..10).map(|x| x as f64).collect()).unwrap();

    arr.mapv_inplace(f);

    println!("{:?}", arr);
    let a = Value::new(34.0);
    let b = Value::new(23.0);
    let c = Value::new(88.0);
    let e = a.clone()+b.clone();
    let d = e.clone()-c.clone();
    let f = Value::new(2.0);
    let L = d.clone()*f.clone();

    a.init();b.init();c.init();d.init();e.init();f.init();
    println!("{:?}", L);
}

fn f(x: f64) -> f64 {
    3.0 * x.powf(2.0) - 4.0 * x + 2.0
}