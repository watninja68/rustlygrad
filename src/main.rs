extern crate ndarray;
extern crate graphviz_rust;
use ndarray::*;
use std::process::Output;
use std::rc::Rc;
use std::collections::{HashSet,  VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::*; 
use graphviz_rust::cmd::{CommandArg, Format};
use graphviz_rust::dot_structures::*;
use graphviz_rust::dot_generator::*;
use graphviz_rust::exec;
use graphviz_rust::attributes::{EdgeAttributes, NodeAttributes};
use crate::graphviz_rust::printer::DotPrinter;
use graphviz_rust::printer::PrinterContext;

#[derive(Debug, Clone)]
struct Value {
    data: f64,
    _op: char,
    grad: f64,
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
            _op: ' ',
            grad: 0.0,
        }
    }

    pub fn tanh(self) -> Value {
        let x = self.data;
        let mut new_prev = self._prev.clone();
        new_prev.insert(Rc::new(self.clone()));
        let temp = ((2.0 * x - 1.0).exp()) / ((2.0 * x + 1.0).exp());
        Value {
            data: temp,
            _op: 'T',
            _prev: new_prev,
            grad: self.grad,
        }
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
            _op: '+',
            grad: self.grad,
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
            _op: '-',
            grad: self.grad,
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
            _op: '*',
            grad: self.grad * other.data,
        }
    }
}
impl  Value {
 

    fn _backward(&mut self) {
        let mut _self = (*(self));
        let grad = _self.grad;
        if let op = &_self._op {
            match op {
               '+' => {
                    // for addition, let them grads flow
                    for child in &mut _self._prev.iter() {
                        child.grad = child.grad + (grad);
                    }
                }
                '*' => {
                    assert!(_self._prev.len() == 2);

                    // for multiplications, it's grad * the_other_guy

                    let grad = _self.grad;
                    let d0 = _self._prev[0].data();
                    let d1 = _self._prev[1].data();

                    _self._prev[0].set_grad(grad * d1);
                    _self._prev[1].set_grad(grad * d0);
                }
                'T' => {
                    assert!(_self._prev.len() == 1);

                    let grad = _self.grad;
                    let d = _self._prev[0].data();

                    // derivative of tanh: (1-tanh ^ 2)
                    _self._prev[0].set_grad(
                        grad * (1.0_f64 - ((2.0_f64 * d - 1.0_f64).exp()) / ((2.0_f64 * d + 1.0_f64).exp()) * ((2.0_f64 * d - 1.0_f64).exp()) / ((2.0_f64 * d + 1.0_f64).exp()))
                    );
                                    }
                // '^' => {
                //     assert!(_self._prev.len() == 1);

                //     // derivate of x^n = n * x ^ (n-1)

                //     let grad = _self.grad;
                //     let d = _self._prev[0].data();
                //     let power_n = *n;

                //     _self._prev[0].set_grad(grad * (f32::from(power_n) * f32::powi(d, (power_n - 1).into())));
                // }
            }

            for child in  _self._prev {
                (*child)._backward();
            }
        }
    }
}

fn trace(root: Rc<Value>) -> (HashSet<Rc<Value>>, HashSet<(Rc<Value>, Rc<Value>)>) {
    let mut nodes: HashSet<Rc<Value>> = HashSet::new();
    let mut edges: HashSet<(Rc<Value>, Rc<Value>)> = HashSet::new();
    let mut queue: VecDeque<Rc<Value>> = VecDeque::new();

    queue.push_back(Rc::clone(&root));

    while let Some(v) = queue.pop_front() {
        if !nodes.contains(&v) {
            nodes.insert(Rc::clone(&v));
            for child in &v._prev {
                edges.insert((Rc::clone(child), Rc::clone(&v)));
                queue.push_back(Rc::clone(child));
            }
        }
    }

    (nodes, edges)
}

fn sanitize(s: &str) -> String {
    s.replace('*', "_star_")
        .replace('+', "_plus_")
        .replace('-', "_minus_")
        .replace('T', "_tanh_")
}

fn draw_dot(root: Rc<Value>) {
    let mut dot = graph!(id!("id"));
    let (nodes, edges) = trace(root);

    for a in &nodes {
        let node_id = sanitize(&a.data.to_string());
        dot.add_stmt(stmt!(node!(node_id.clone())));
        if a._op == '+' || a._op == '-' || a._op == '*' || a._op == 'T' {
            let op_node = sanitize(&(a.data.to_string() + &a._op.to_string()));
            dot.add_stmt(stmt!(node!(op_node.clone())));
            dot.add_stmt(stmt!(edge!(node_id!(node_id.clone()) => node_id!(op_node))));
        }
    }

    for (n1, n2) in &edges {
        let n1_id = sanitize(&n1.data.to_string());
        let n2_id = sanitize(&(n2.data.to_string() + &n2._op.to_string()));
        dot.add_stmt(stmt!(edge!(node_id!(n1_id) => node_id!(n2_id))));
    }

    println!("{}", dot.print(&mut PrinterContext::default()));
    let p = "1.svg";
    let out = exec(dot, &mut PrinterContext::default(), vec![
        CommandArg::Format(Format::Svg),
        CommandArg::Output(p.to_string())
    ]).unwrap();

    assert_eq!("", out);
}

fn main() {
    let mut arr = Array2::from_shape_vec((3, 3), (1..10).map(|x| x as f64).collect()).unwrap();
    arr.mapv_inplace(f);
    println!("{:?}", arr);

    let a = Value::new(34.0);
    let b = Value::new(23.0);
    let c = Value::new(88.0);
    let e = a.clone() + b.clone();
    let d = e.clone() - c.clone();
    let f = Value::new(2.0);
    let mut l = d.clone() * f.clone();
    l = l.tanh();

    // println!("{:?}", l);
    // draw_dot(l.into());
}

fn f(x: f64) -> f64 {
    3.0 * x.powf(2.0) - 4.0 * x + 2.0
}
