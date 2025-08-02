#!/usr/bin/env python3
"""
Zoo Placement Algorithm using Real Resource Data from User
Level 1: Basic Placement for 50x50 grid
Available resources: [1, 3, 4, 6, 9, 10, 11, 14, 15, 20, 21]
"""

import json
import random
import time

class RealZooPlacement:
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.grid = [[1 for _ in range(grid_size)] for _ in range(grid_size)]
        self.occupied = [[False for _ in range(grid_size)] for _ in range(grid_size)]
        self.resources = []
        self.resource_counts = {}
        self.resource_placements = {}
        
    def load_level1_resources(self):
        """Load Level 1 resources with actual data from user"""
        
        # Real Level 1 resource definitions (excluding pathways for placement)
        level1_data = [
            {
                "resource_id": 3,
                "name": "Coffee shop",
                "bounding_box": 5,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 3], [2, 4]]}
                ]
            },
            {
                "resource_id": 4,
                "name": "Bathroom",
                "bounding_box": 8,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [0, 2], [0, 5], [0, 6], [0, 7], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 0], [2, 1], [2, 2], [2, 5], [2, 6], [2, 7]]}
                ]
            },
            {
                "resource_id": 6,
                "name": "Gift shop",
                "bounding_box": 9,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]]}
                ]
            },
            {
                "resource_id": 9,
                "name": "Guineafowl",
                "bounding_box": 3,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]}
                ]
            },
            {
                "resource_id": 10,
                "name": "Spotted Eagle Owl",
                "bounding_box": 4,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]}
                ]
            },
            {
                "resource_id": 11,
                "name": "Tortoise",
                "bounding_box": 4,
                "orientations": [
                    {"cells": [[0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [3, 0], [3, 1]]}
                ]
            },
            {
                "resource_id": 14,
                "name": "Boomslang",
                "bounding_box": 4,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]]}
                ]
            },
            {
                "resource_id": 15,
                "name": "Springbok",
                "bounding_box": 8,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 0], [2, 1], [2, 6], [2, 7], [3, 0], [3, 1], [3, 6], [3, 7]]}
                ]
            },
            {
                "resource_id": 20,
                "name": "Warthog",
                "bounding_box": 8,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [0, 2], [0, 4], [0, 5], [0, 6], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 2], [2, 5], [2, 6], [2, 7]]}
                ]
            },
            {
                "resource_id": 21,
                "name": "Fox",
                "bounding_box": 6,
                "orientations": [
                    {"cells": [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [5, 0], [5, 1]]}
                ]
            }
        ]
        
        self.resources = level1_data
        
        # Initialize tracking
        for resource in self.resources:
            rid = resource["resource_id"]
            self.resource_counts[rid] = 0
            self.resource_placements[rid] = 0
            
        print(f"Loaded {len(self.resources)} Level 1 resources")
        
    def can_place(self, resource, orientation, top, left):
        """Check if resource can be placed at position"""
        cells = orientation["cells"]
        resource_id = resource["resource_id"]
        
        # Check bounds and occupation
        for dr, dc in cells:
            r, c = top + dr, left + dc
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                return False
            if self.occupied[r][c]:
                return False
        
        # Check 1-block gap for same resource
        for dr, dc in cells:
            r, c = top + dr, left + dc
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.grid[nr][nc] == resource_id:
                            return False
        
        return True
    
    def place_resource(self, resource, orientation, top, left):
        """Place resource on grid"""
        resource_id = resource["resource_id"]
        cells = orientation["cells"]
        
        for dr, dc in cells:
            r, c = top + dr, left + dc
            self.grid[r][c] = resource_id
            self.occupied[r][c] = True
        
        self.resource_counts[resource_id] += len(cells)
        self.resource_placements[resource_id] += 1
    
    def optimize_placement(self):
        """Main optimization loop"""
        print("Starting optimization...")
        
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            # Prioritize resources that haven't been placed yet
            unplaced = [r for r in self.resources if self.resource_placements[r["resource_id"]] == 0]
            if unplaced:
                resource = random.choice(unplaced)
            else:
                resource = random.choice(self.resources)
            
            orientation = random.choice(resource["orientations"])
            
            # Try random positions
            for _ in range(50):
                bbox = resource["bounding_box"]
                if bbox >= self.grid_size:
                    continue
                
                top = random.randint(0, self.grid_size - bbox)
                left = random.randint(0, self.grid_size - bbox)
                
                if self.can_place(resource, orientation, top, left):
                    self.place_resource(resource, orientation, top, left)
                    break
            
            if attempt % 100 == 0:
                diversity = len([rid for rid, count in self.resource_placements.items() if count > 0])
                total_placed = sum(self.resource_placements.values())
                print(f"Attempt {attempt}: {total_placed} placements, {diversity} unique resources")
    
    def get_statistics(self):
        """Get final statistics"""
        total_cells = sum(self.resource_counts.values())
        total_placements = sum(self.resource_placements.values())
        unique_resources = len([rid for rid, count in self.resource_placements.items() if count > 0])
        utilization = (total_cells / (self.grid_size * self.grid_size)) * 100
        
        return {
            "total_cells_placed": total_cells,
            "total_placements": total_placements,
            "unique_resources_used": unique_resources,
            "grid_utilization_percent": round(utilization, 2),
            "resource_breakdown": dict(self.resource_placements)
        }
    
    def save_solution(self, filename="real_zoo_solution.txt"):
        """Save solution"""
        solution = {
            "level": 1,
            "zoo_size": f"{self.grid_size}x{self.grid_size}",
            "resources": [1, 3, 4, 6, 9, 10, 11, 14, 15, 20, 21],
            "zoo": self.grid,
            "statistics": self.get_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(solution, f, indent=2)
        
        print(f"Solution saved to {filename}")

def main():
    print("=== Zoo Placement with Real Resource Data ===")
    print("Level 1: Available resources [1, 3, 4, 6, 9, 10, 11, 14, 15, 20, 21]")
    
    random.seed(42)
    
    optimizer = RealZooPlacement()
    optimizer.load_level1_resources()
    
    start_time = time.time()
    optimizer.optimize_placement()
    end_time = time.time()
    
    stats = optimizer.get_statistics()
    
    print("\n=== FINAL RESULTS ===")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Total cells placed: {stats['total_cells_placed']}")
    print(f"Total placements: {stats['total_placements']}")
    print(f"Unique resources used: {stats['unique_resources_used']}/10")
    print(f"Grid utilization: {stats['grid_utilization_percent']}%")
    print()
    print("Resource breakdown:")
    for resource_id, count in stats['resource_breakdown'].items():
        if count > 0:
            resource_name = next(r["name"] for r in optimizer.resources if r["resource_id"] == resource_id)
            print(f"  {resource_name} (ID {resource_id}): {count} placements")
    
    optimizer.save_solution()
    print("\nOptimization complete!")

if __name__ == "__main__":
    main()
