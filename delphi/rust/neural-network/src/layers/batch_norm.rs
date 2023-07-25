// use crate::{
//     layers::LayerDims,
//     tensors::{Input, Output},
// };
// // copied use statements from non_linear.rs (don't know if
// // necessary or not)
// use num_traits::{One, Zero};
// use std::{
//     marker::PhantomData,
//     ops::{AddAssign, Mul, MulAssign}
// };

// // is this really even necessary? There's only one 
// // variant here
// #[derive(Debug)]
// pub struct BatchNormLayer<F, C> {
//     // I've forgotten what this Rust notation means
//     // review it in the Rust docs
//     pub eval_method: crate::EvalMethod,
// }

// impl BatchNormLayer {
//     // dimensions() method returns a LayerDims struct
//     pub fn dimensions(&self) -> LayerDims {
//         match self {
//             BatchNorm(dims) => *dims,
//         }
//     }
//     // .input_dimensions() is called a LayerDims struct in the 
//     // mod.rs file (basically a wrapper function)
//     pub fn input_dimensions(&self) -> (usize, usize, usize, usize) {
//         self.dimensions().input_dimensions()
//     }
//     // .output_dimensions() is also called on the output dimensions
//     pub fn output_dimensions(&self) -> (usize, usize, usize, usize) {
//         self.dimensions().output_dimensions()
//     }
// }

// // still not sure what the num_trait crate needs to be used for
// // it's not obvious to me how to use or what to use it for

// // crate::EvalMethod is stored within the lib.rs file (project
// // crate for neural-network lib project)

// // the EvalMethod enum just selects whether you use torch device
// // or a naive method to evaluate the

// // Q: Do we really need PartialOrd<C>? 

// // code for Output::zeros is contained in the neural-network/src/tensors
// impl<F, C> BatchNormLayer<F, C>
// where 
//     F: Zero + Copy + Mul<C, Output=F> + AddAssign,
//     C: Copy + Into<F>,
// {
//     // no need for calculation of the output size
//     pub fn naive_batch_norm(&self, input: &Input<F>, out: &mut Output<F>) {
        
//     }
// }