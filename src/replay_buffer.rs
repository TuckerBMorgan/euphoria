use ndarray::prelude::*;
use rand::prelude::*;


#[derive(Clone, Copy, Default)]
pub struct GenericMemory<S: Default + Clone, A: Default + Clone, R: Default + Clone, D: Default + Clone> {
    state: S,
    action: A,
    reward: R,
    next_state: S,
    done: D,
}

impl<S: Default + Clone, A: Default + Clone, R: Default + Clone, D: Default + Clone> GenericMemory<S, A, R, D> {
    pub fn new(state: S, action: A, reward: R, next_state: S, done: D) -> GenericMemory<S, A, R, D> {
        GenericMemory {
            state,
            action,
            reward,
            next_state,
            done
        }
    }
}

pub struct GenericMemoryBuffer<S: Default + Clone, A: Default + Clone, R: Default + Clone, D: Default + Clone> {
    memories: Vec<GenericMemory<S, A, R, D>>,
    capacity: usize,
    number_of_added_memories: usize
}

impl<S: Default + Clone, A: Default + Clone, R: Default + Clone, D: Default + Clone> GenericMemoryBuffer<S, A, R, D> {
    pub fn new(capacity: usize) -> GenericMemoryBuffer<S, A, R, D> {
        GenericMemoryBuffer {
            memories: vec![Default::default(); capacity],
            capacity,
            number_of_added_memories: 0
        }
    }

    pub fn add_memory(&mut self, memory: GenericMemory<S, A, R, D>) {
        self.memories[self.number_of_added_memories % self.capacity] = memory; 
    }

    pub fn sample_batch(&self, sample_size: usize) -> Vec<&GenericMemory<S, A, R, D>> {
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
}



pub struct Memory {
    states: Array2<f32>,
    actions: Array2<f32>,
    reward: f32,
    next_state: Array2<f32>,
    terminal: bool
}

pub struct ReplayBuffer {
    states: Vec<Array2<f32>>,
    actions: Vec<usize>,
    reward: Vec<f32>,
    next_state: Vec<Array2<f32>>,
    terminal: Vec<bool>,
    pub current_number_of_memories: usize,
    buffer_size: usize
}

impl ReplayBuffer {
    pub fn new(size: usize) -> ReplayBuffer {
        ReplayBuffer {
            states: vec![Array2::zeros((1, 16)); size],
            actions: vec![0; size],
            reward: vec![0.0f32; size],
            next_state: vec![Array2::zeros((1, 16)); size],
            terminal: vec![false; size],
            current_number_of_memories: 0,
            buffer_size: size
        }
    }

    pub fn add_memory(&mut self, state: Array2<f32>,
                                 action: usize,
                                 reward: f32,
                                 next_state: Array2<f32>,
                                 terminal: bool) {
        let next_index = self.current_number_of_memories % self.buffer_size;
        self.states[next_index] = state;
        self.actions[next_index] = action;
        self.reward[next_index] = reward;
        self.next_state[next_index] = next_state;
        self.terminal[next_index] = terminal;

        self.current_number_of_memories += 1;
    }

    pub fn sample_batch(&self, number_of_samples: usize) -> Vec<(&Array2<f32>, &usize, &f32, &Array2<f32>, &bool)>{
        let mut rng = rand::thread_rng();
        let mut terrible = vec![];

        for _ in 0..number_of_samples {
            terrible.push(rng.gen_range(0, self.current_number_of_memories.min(self.buffer_size)));            
        }
        
        let mut sarsas = vec![];
        for i in terrible {
            let memory = (&self.states[i], &self.actions[i], &self.reward[i], &self.next_state[i], &self.terminal[i]);
            sarsas.push(memory);
        }
        return sarsas;
    }
}