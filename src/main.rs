use rand::Rng;

#[derive(Debug)]
enum ActivationFunction {
    Sigmoid,
    Relu,
}

impl ActivationFunction {
    fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Relu => if x > 0.0 { x } else { 0.0 },
        }
    }

    fn apply_vector(&self, v: &Vec<f64>) -> Vec<f64> {
        v.iter().map(|&x| self.apply(x)).collect()
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let sig = self.apply(x);
                sig * (1.0 - sig)
            },
            ActivationFunction::Relu => {
                if x > 0.0 { 1.0 } else { 0.0 }
            },
        }
    }

    fn derivative_vector(&self, v: &Vec<f64>) -> Vec<f64> {
        v.iter().map(|&x| self.derivative(x)).collect()
    }
}

#[derive(Debug)]
enum Layer {
    Input {
        shape: (usize, usize),
    },
    Dense {
        units: usize,
        activation_function: ActivationFunction,
    },
}

#[derive(Debug)]
struct NeuralNetwork {
    layers: Vec<Layer>,
    weights: Vec<Vec<Vec<f64>>>,  
    biases: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    fn new(layers: Vec<Layer>) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut prev_units = 0;

        for i in &layers {
            match i {
                Layer::Input { shape } => {
                    let (m, _n) = shape;
                    prev_units = *m;
                },
                Layer::Dense { units, activation_function: _ } => {
                    weights.push(NeuralNetwork::init_weights(*units, prev_units));
                    biases.push(vec![0.0; *units]);
                    prev_units = *units;
                }
            }
        }
        NeuralNetwork {
            layers,
            weights,
            biases,
        }
    }

    fn init_weights(m: usize, n: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        (0..m).map(|_| {
            (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect()
        }).collect()
    }

     fn forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut activations = input;
        let mut all_activations = vec![activations.clone()];
        let mut all_zs = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Input { .. } => continue,
                Layer::Dense { activation_function, .. } => {
                    let z: Vec<f64> = (0..self.weights[i-1].len()).map(|j| {
                        self.weights[i-1][j].iter().zip(&activations).map(|(w, a)| w * a).sum::<f64>() + self.biases[i-1][j]
                    }).collect();
                    all_zs.push(z.clone());
                    activations = activation_function.apply_vector(&z);
                    all_activations.push(activations.clone());
                }
            }
        }

        activations
    }

    fn backward(&mut self, input: Vec<f64>, target: Vec<f64>, learning_rate: f64) {
        let mut activations = input.clone();
        let mut all_activations = vec![activations.clone()];
        let mut all_zs = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Input { .. } => continue, 
                Layer::Dense { activation_function, .. } => {
                    let z: Vec<f64> = (0..self.weights[i-1].len()).map(|j| {
                        self.weights[i-1][j].iter().zip(&activations).map(|(w, a)| w * a).sum::<f64>() + self.biases[i-1][j]
                    }).collect();
                    all_zs.push(z.clone());
                    activations = activation_function.apply_vector(&z);
                    all_activations.push(activations.clone());
                }
            }
        }

        let output_activations = all_activations.last().unwrap();
        let mut delta: Vec<f64> = output_activations.iter().zip(target.iter()).map(|(&o, &t)| o - t).collect();

        for i in (1..self.layers.len()).rev() {
            match &self.layers[i] {
                Layer::Dense { activation_function, .. } => {
                    let z = &all_zs[i - 1];
                    let sp = activation_function.derivative_vector(z);
                    delta = delta.iter().zip(sp.iter()).map(|(&d, &sp)| d * sp).collect();

                    let prev_activations = &all_activations[i - 1];
                    for j in 0..self.weights[i-1].len() {
                        for k in 0..self.weights[i-1][j].len() {
                            self.weights[i-1][j][k] -= learning_rate * delta[j] * prev_activations[k];
                        }
                        self.biases[i-1][j] -= learning_rate * delta[j];
                    }
                    if i > 1 {
                        delta = (0..self.weights[i-1][0].len()).map(|k| {
                            (0..self.weights[i-1].len()-1).map(|j| self.weights[i-1][j][k] * delta[j]).sum::<f64>()
                        }).collect();
                    }
                },
                _ => {}
            }
        }
    }
    fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.backward(input.clone(), target.clone(), learning_rate);
            }
        }
    }

    fn predict(&mut self, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        inputs.iter().map(|x| self.forward(x.to_vec())).collect()
    }
}

fn main() {
    let layers = vec![
        Layer::Input { shape: (2, 1) }, 
        Layer::Dense { units: 32, activation_function: ActivationFunction::Relu }, 
        Layer::Dense { units: 1, activation_function: ActivationFunction::Sigmoid }
    ];
        
let features: Vec<Vec<f64>> = vec![
    vec![0.4967, -0.1383],
    vec![0.6477, 1.5230],
    vec![-0.2341, -0.2341],
    vec![1.5792, 0.7674],
    vec![0.5426, -0.4616],
    vec![0.2419, -1.9133],
    vec![1.4656, -0.2258],
    vec![0.0675, -1.4247],
    vec![-0.5443, 0.1109],
    vec![1.1504, 0.3757],
    vec![3.3757, 4.1145],
    vec![3.6774, 3.1915],
    vec![3.2575, 2.2150],
    vec![3.5145, 2.5868],
    vec![2.6638, 3.4244],
    vec![4.0247, 3.9265],
    vec![2.8658, 3.1945],
    vec![3.6972, 3.7902],
    vec![2.7328, 3.3020],
    vec![2.8640, 3.8874],
];

let targets: Vec<Vec<f64>> = vec![
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![0.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
    vec![1.0],
];
    let mut nn = NeuralNetwork::new(layers);
    nn.train(features, targets, 1000, 0.01);
    println!("{:?}", nn.predict(&vec![vec![1.2342, -1.3213]]));
}
