use ndarray::prelude::*;
use tsuga::prelude::*;
use crate::replay_buffer::*;

pub struct TargetNetworkDQN {
    target_network: FullyConnectedNetwork,
    live_network: FullyConnectedNetwork,
    memory_buffer: GenericMemoryBuffer
    //optimize network
}

impl TargetNetworkDQN {
    pub fn new(live_network: FullyConnectedNetwork, target_network: FullyConnectedNetwork) -> TargetNetworkDQN {
        TargetNetworkDQN {
            target_network,
            live_network,
            memory_buffer: GenericMemoryBuffer::new(10_000)
        }
    }

    pub fn add_memory(&mut self, memory: GenericMemory) {
        self.memory_buffer.add_memory(memory);
    }

    pub fn predict(&mut self, input: Array2<f32>) -> Array2<f32> {
        return self.live_network.predict(input);
    }

    pub fn number_of_collected_memories(&self) -> usize {
        self.memory_buffer.number_of_samples_collected()
    }

    pub fn optimize_models(&mut self, batch_size: usize) {
        let reward_discount_factor = 0.995f32;
        let memories = self.memory_buffer.sample_batch(batch_size);
        for memory in memories {
            
            let mut rewards_of_actions = self.live_network.predict(memory.state.clone());
            
            let rewards_of_possible_next_state = self.target_network.predict(memory.next_state.clone());

            if memory.done == true {
                rewards_of_actions[[0, memory.action]] = memory.reward;
            } else {
                rewards_of_actions[[0, memory.action]] = memory.reward + (reward_discount_factor * rewards_of_possible_next_state.iter().max_by(|x, y|x.partial_cmp(&y).unwrap()).unwrap());
            }
            self.live_network.single_training_batch(memory.state.clone(), rewards_of_actions, 1);
        }

    }

    pub fn set_target_network_to_q_network(&mut self) {
        self.live_network.blind_copy(&mut self.target_network);
    }
}