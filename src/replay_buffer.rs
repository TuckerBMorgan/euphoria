use ndarray::prelude::*;
use rand::prelude::*;


#[derive(Clone, Default)]
pub struct GenericMemory {
    pub state: Array2<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Array2<f32>,
    pub done: bool,
}

impl GenericMemory {
    pub fn new(state: Array2<f32>, action:usize, reward: f32, next_state: Array2<f32>, done: bool) -> GenericMemory {
        GenericMemory {
            state,
            action,
            reward,
            next_state,
            done
        }
    }
}

pub struct GenericMemoryBuffer {
    memories: Vec<GenericMemory>,
    capacity: usize,
    number_of_added_memories: usize
}

impl GenericMemoryBuffer {
    pub fn new(capacity: usize) -> GenericMemoryBuffer {
        GenericMemoryBuffer {
            memories: vec![Default::default(); capacity],
            capacity,
            number_of_added_memories: 0
        }
    }

    pub fn add_memory(&mut self, memory: GenericMemory) {
        self.memories[self.number_of_added_memories % self.capacity] = memory; 
        self.number_of_added_memories += 1;
    }

    pub fn sample_batch(&self, sample_size: usize) -> Vec<&GenericMemory> {
        let mut rng = rand::thread_rng();
        let mut terrible = vec![];

        for _ in 0..sample_size {
            terrible.push(rng.gen_range(0, self.number_of_added_memories.min(self.capacity)));            
        }
        let mut memories = vec![];
        for index in terrible {
            memories.push(&self.memories[index]);
        }
        return memories;
    }


    pub fn number_of_samples_collected(&self) -> usize {
        return self.number_of_added_memories;
    }
}