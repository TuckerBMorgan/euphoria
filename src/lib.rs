
mod replay_buffer;
mod target_network_dqn;
mod discrete_space_network;

/// Contains all the necessary imports for building and training a basic neural network
pub mod prelude {
    pub use crate::replay_buffer::*;
    pub use crate::target_network_dqn::*;
    pub use crate::discrete_space_network::*;
}
