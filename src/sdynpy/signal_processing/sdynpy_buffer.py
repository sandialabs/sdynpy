# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 12:34:27 2025

@author: dprohe
"""
import numpy as np

class CircularBufferWithOverlap:
    def __init__(self, buffer_size, block_size, overlap_size, dtype = 'float', data_shape = ()):
        """
        Initialize the circular buffer.
        
        Parameters:
        - buffer_size: Total size of the circular buffer.
        - block_size: Number of samples written in each block.
        - overlap_size: Number of samples from the previous block to include in the read.
        """
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.overlap_size = overlap_size
        self.buffer = np.zeros(tuple(data_shape) + (buffer_size,), dtype=dtype)  # Initialize buffer with zeros
        self.buffer_read = np.ones((buffer_size,),dtype=bool)
        self.write_index = 0  # Index where the next block will be written
        self.read_index = 0 # Index where the next block will be read from
    
    def report_buffer_state(self):
        read_samples = np.sum(self.buffer_read)
        write_samples = self.buffer_size - read_samples
        print(f'{read_samples} of {self.buffer_size} have been read')
        print(f'{write_samples} of {self.buffer_size} have been written but not read')

    def write_get_data(self,data, read_remaining = False):
        """
        Writes a block of data and then returns a block if available

        Parameters:
        - data: Array to write to the buffer.
        """
        self.write(data)
        try:
            return self.read(read_remaining)
        except ValueError:
            return None
    
    def write(self, data):
        """
        Write a block of data to the circular buffer.
        
        Parameters:
        - data: Array to write to the buffer.
        """
        # Compute the end index for the write operation
        indices = np.arange(self.write_index,self.write_index+data.shape[-1]+self.overlap_size) % self.buffer_size

        if np.any(~self.buffer_read[indices]):
            raise ValueError('Overwriting data on buffer that has not been read.  Read data before writing again.')

        self.buffer[...,indices[:None if self.overlap_size == 0 else -self.overlap_size]] = data
        self.buffer_read[indices[:None if self.overlap_size == 0 else -self.overlap_size]] = False
        
        # Update the write index
        self.write_index = (self.write_index + data.shape[-1]) % self.buffer_size
        
        # print(self.buffer)
        # print(self.buffer_read)
        
    def read(self, read_remaining = False):
        indices = np.arange(self.read_index - self.overlap_size, self.read_index + self.block_size) % self.buffer_size
        if read_remaining:
            # Pick out just the indices that are ok to read
            # print('Reading Remaining:')
            # print(f"{indices.copy()=}")
            indices = np.concatenate((indices[:self.overlap_size],indices[self.overlap_size:][~self.buffer_read[indices[self.overlap_size:]]]))
            # print(f"{indices.copy()=}")
        if np.any(self.buffer_read[indices[self.overlap_size:]]):
            raise ValueError('Data would be read multiple times.  Write data before reading again.')
        return_data = self.buffer[...,indices]
        self.buffer_read[indices[self.overlap_size:]] = True
        self.read_index = (self.read_index + (return_data.shape[-1]-self.overlap_size)) % self.buffer_size
        # print(self.buffer)
        # print(self.buffer_read)
        return return_data
