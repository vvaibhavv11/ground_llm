use crate::matrix::Matrix;

pub struct Mlp {
    pub n_hidden: usize,
    pub n_input_node: usize,
    pub weights: Vec<Matrix>,
    // pub node_data: Vec<Matrix>,
}

impl Mlp {
    pub fn new(n_input_node: usize, n_hidden: usize) -> Mlp {
        let mut weights = Vec::new();
        let w1_input_h1_weight = Matrix::random(n_input_node, 1380);
        let w2_input_h1_weight = Matrix::random(n_input_node, 1380);
        let w3_hn_output_weight = Matrix::random(1380, n_input_node);
        weights.push(w1_input_h1_weight);
        weights.push(w2_input_h1_weight);
        weights.push(w3_hn_output_weight);
        // let mut node_data = Vec::new();
        // let mut input_data_matrix = Matrix::new(1, n_input_node);
        // let mut hidden_layar_matrix = Matrix::new(1, 4 * n_input_node);
        // let mut output_layar_matrix = Matrix::new(1, 4 * n_input_node);
        // node_data.push(input_data_matrix);
        // node_data.push(hidden_layar_matrix);
        // node_data.push(output_layar_matrix);
        Mlp {
            n_hidden,
            n_input_node,
            weights,
        }
    }

    pub fn feedforward(&mut self, input_nods: Matrix) {
        let mut a = input_nods.mul_transpose(&self.weights[0].transpose());
        let b = input_nods.mul_transpose(&self.weights[1].transpose());
        a.swish();
        a.elem_mul(&b);
        let y = a.mul_transpose(&self.weights[2].transpose());
    }
}
