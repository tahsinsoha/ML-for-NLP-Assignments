#!/usr/bin/env python3
"""
IBM Model 1 training using EM algorithm with checkpointing
"""

import os
import pickle
from collections import defaultdict
from typing import List, Tuple

from config import (
    NUM_EM_ITERATIONS, PREPROCESSED_DATA_FILE, 
    CHECKPOINT_DIR, MODEL_CHECKPOINT_PREFIX, FINAL_MODEL_FILE
)


class IBMModel1:
    """
    IBM Model 1 for word alignment.
    Estimates P(e|f) using EM algorithm.
    """
    
    def __init__(self):
        self.t = defaultdict(lambda: defaultdict(float))
        self.source_vocab = set()
        self.target_vocab = set()
        self.uniform_prob = 0.0
        self.first_iteration = True
        self.current_iteration = 0
        
    def get_prob(self, e: str, f: str) -> float:
        if self.first_iteration:
            return self.uniform_prob
        return self.t[e][f]
    
    def build_vocabularies(self, parallel_data: List[Tuple[List[str], List[str]]]):
        print("Building vocabularies...")
        for f_sent, e_sent in parallel_data:
            self.source_vocab.update(f_sent)
            self.target_vocab.update(e_sent)
        
        self.source_vocab.add("NULL")
        self.uniform_prob = 1.0 / len(self.target_vocab)
        
        print(f"Source vocab: {len(self.source_vocab)}, Target vocab: {len(self.target_vocab)}")
        print(f"Initial t(e|f) = 1/{len(self.target_vocab)} = {self.uniform_prob:.10f}")
    
    def run_em_iteration(self, parallel_data: List[Tuple[List[str], List[str]]]):
        """Run one EM iteration."""
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        
        for f_sent, e_sent in parallel_data:
            f_sent_with_null = ["NULL"] + f_sent
            
            for e in e_sent:
                s_total = sum(self.get_prob(e, f) for f in f_sent_with_null)
                
                if s_total > 0:
                    for f in f_sent_with_null:
                        delta = self.get_prob(e, f) / s_total
                        count[e][f] += delta
                        total[f] += delta
        
        self.t = defaultdict(lambda: defaultdict(float))
        for f in total:
            if total[f] > 0:
                for e in count:
                    if count[e][f] > 0:
                        self.t[e][f] = count[e][f] / total[f]
        
        self.first_iteration = False
    
    def train(self, parallel_data: List[Tuple[List[str], List[str]]], 
              num_iterations: int = 5, save_checkpoints: bool = True):
        
        if len(self.source_vocab) == 0:
            self.build_vocabularies(parallel_data)
        
        if save_checkpoints:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        print(f"\nRunning {num_iterations} EM iterations...")
        
        for iteration in range(self.current_iteration, num_iterations):
            self.run_em_iteration(parallel_data)
            self.current_iteration = iteration + 1
            
            print(f"  Iteration {iteration + 1}/{num_iterations}")
            
            if save_checkpoints:
                checkpoint_file = f"{MODEL_CHECKPOINT_PREFIX}_{iteration + 1}.pkl"
                self.save(checkpoint_file)
        
        if save_checkpoints:
            self.save(FINAL_MODEL_FILE)
            print(f"Model saved to {FINAL_MODEL_FILE}")
    
    def save(self, filepath: str):
        model_data = {
            't': {e: dict(f_probs) for e, f_probs in self.t.items()},
            'source_vocab': self.source_vocab,
            'target_vocab': self.target_vocab,
            'uniform_prob': self.uniform_prob,
            'first_iteration': self.first_iteration,
            'current_iteration': self.current_iteration
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'IBMModel1':
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.source_vocab = model_data['source_vocab']
        model.target_vocab = model_data['target_vocab']
        model.uniform_prob = model_data['uniform_prob']
        model.first_iteration = model_data['first_iteration']
        model.current_iteration = model_data['current_iteration']
        
        for e, f_probs in model_data['t'].items():
            for f, prob in f_probs.items():
                model.t[e][f] = prob
        
        return model


def load_preprocessed_data(filepath: str) -> List[Tuple[List[str], List[str]]]:
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    print("Training IBM Model 1\n")
    
    print(f"Loading data from {PREPROCESSED_DATA_FILE}...")
    parallel_data = load_preprocessed_data(PREPROCESSED_DATA_FILE)
    print(f"Loaded {len(parallel_data)} sentence pairs.")
    
    model = IBMModel1()
    
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) 
                            if f.startswith("model_iter_") and f.endswith(".pkl")])
        if checkpoints:
            latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            print(f"Resuming from {latest}")
            model = IBMModel1.load(latest)
    
    model.train(parallel_data, num_iterations=NUM_EM_ITERATIONS, save_checkpoints=True)


if __name__ == "__main__":
    main()
