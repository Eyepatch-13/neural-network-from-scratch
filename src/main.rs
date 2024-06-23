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
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Input { .. } => continue, 
                Layer::Dense { activation_function, .. } => {
                    let z: Vec<f64> = (0..self.weights[i-1].len()).map(|j| {
                        self.weights[i-1][j].iter().zip(&activations).map(|(w, a)| w * a).sum::<f64>() + self.biases[i-1][j]
                    }).collect();
                    activations = activation_function.apply_vector(&z);
                }
            }
        }
        activations
    }
}

fn main() {
    let layers = vec![
        Layer::Input { shape: (32, 1) }, 
        Layer::Dense { units: 16, activation_function: ActivationFunction::Relu }, 
        Layer::Dense { units: 1, activation_function: ActivationFunction::Sigmoid }
    ];

    let nn = NeuralNetwork::new(layers);
    println!("{:?}", nn.forward(vec![1.0;32]));
}
