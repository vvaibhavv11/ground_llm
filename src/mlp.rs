use crate::matrix::{self, Matrix};

pub struct Mlp {
    pub n_hidden: usize,
    pub n_input_perceptron: usize,

    // W1, W2, W3
    pub weights: Vec<Matrix>,

    // activations for backprop
    // [input, hidden, output]
    pub node_data: Vec<Matrix>,
}

impl Mlp {
    pub fn new(seq_len: usize, n_input_perceptron: usize) -> Mlp {
        let n_hidden = 1380;

        // -----------------------------
        // Weights
        // -----------------------------

        let mut weights = Vec::new();

        // [512, 1380]
        let w1_input_h1_weight = Matrix::random(n_input_perceptron, n_hidden);

        // [512, 1380]
        let w2_input_h1_weight = Matrix::random(n_input_perceptron, n_hidden);

        // [1380, 512]
        let w3_hn_output_weight = Matrix::random(n_hidden, n_input_perceptron);

        weights.push(w1_input_h1_weight);
        weights.push(w2_input_h1_weight);
        weights.push(w3_hn_output_weight);

        // -----------------------------
        // Activations
        // -----------------------------
        //
        // rows    = tokens
        // columns = perceptrons/features
        //
        // [seq_len, hidden_dim]
        //
        // NOT [1, hidden_dim]
        //
        // because transformers process
        // all token vectors simultaneously
        // through the same FFN.
        //
        // Position-wise FFNs apply the same
        // MLP independently to every token
        // position. :contentReference[oaicite:0]{index=0}
        //
        // -----------------------------

        let mut node_data = Vec::new();

        // input activations
        // [seq_len, 512]
        let input_data_matrix = Matrix::new(seq_len, n_input_perceptron);

        // hidden activations
        // [seq_len, 1380]
        let hidden_layer_matrix = Matrix::new(seq_len, n_hidden);

        // output activations
        // [seq_len, 512]
        let output_layer_matrix = Matrix::new(seq_len, n_input_perceptron);

        node_data.push(input_data_matrix);
        node_data.push(hidden_layer_matrix);
        node_data.push(output_layer_matrix);

        Mlp {
            n_hidden,
            n_input_perceptron,
            weights,
            node_data,
        }
    }

    pub fn feedforward(&mut self, input_nods: Matrix) -> Matrix {
        let mut a = input_nods.mul_transpose(&self.weights[0].transpose());
        let b = input_nods.mul_transpose(&self.weights[1].transpose());
        a.swish();
        a.elem_mul(&b);
        a.mul_transpose(&self.weights[2].transpose())
    }
}
