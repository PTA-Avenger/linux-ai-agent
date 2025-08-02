import json
import random
from collections import defaultdict
import os

class EnhancedZooPlacementOptimizer:
    def __init__(self, grid_size=50):
        self.GRID_SIZE = grid_size
        self.grid = [[1 for _ in range(grid_size)] for _ in range(grid_size)]
        self.occupied = [[False for _ in range(grid_size)] for _ in range(grid_size)]
        self.resource_counts = defaultdict(int)
        self.resource_placements = defaultdict(int)  # Track number of individual placements
        self.total_placed = 0
        
    def load_base_zoo(self, filename="1.txt"):
        """Load base zoo from file if it exists"""
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Using empty grid.")
            return
            
        try:
            with open(filename, "r") as f:
                lines = f.readlines()
            
            # Find the base zoo section
            zoo_start_index = next(i for i, line in enumerate(lines) if line.startswith("Base Zoo"))
            zoo_data_lines = lines[zoo_start_index + 1:]
            
            # Parse the grid
            zoo = []
            for line in zoo_data_lines:
                try:
                    row = json.loads(line.strip().rstrip(","))
                    if isinstance(row, list) and len(row) == self.GRID_SIZE:
                        zoo.append(row)
                    if len(zoo) == self.GRID_SIZE:
                        break
                except:
                    continue
            
            if len(zoo) == self.GRID_SIZE:
                self.grid = zoo
                # Update occupied status
                for r in range(self.GRID_SIZE):
                    for c in range(self.GRID_SIZE):
                        if self.grid[r][c] != 1:
                            self.occupied[r][c] = True
                            self.resource_counts[self.grid[r][c]] += 1
                            self.total_placed += 1
                            
        except Exception as e:
            print(f"Error loading base zoo: {e}")
    
    def load_resources(self, filename="resources.json", allowed_ids=None):
        """Load resource definitions from JSON file"""
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Creating sample resources.")
            return self.create_sample_resources(allowed_ids)
            
        try:
            with open(filename, "r") as f:
                resources_data = json.load(f)["resources"]
            
            if allowed_ids:
                allowed_resource_ids = set(allowed_ids)
                resources = [res for res in resources_data 
                           if res["resource_id"] in allowed_resource_ids and res["resource_id"] != 1]
            else:
                resources = [res for res in resources_data if res["resource_id"] != 1]
                
            return resources
        except Exception as e:
            print(f"Error loading resources: {e}")
            return self.create_sample_resources(allowed_ids)
    
    def create_sample_resources(self, allowed_ids=None):
        """Create sample resources for testing when resources.json is not available"""
        if allowed_ids is None:
            allowed_ids = [3, 4, 6, 9, 10, 11, 14, 15, 20, 21]
            
        sample_resources = []
        for resource_id in allowed_ids:
            if resource_id == 1:  # Skip pathways
                continue
                
            # Create varied resource shapes and sizes
            if resource_id == 3:
                resource = {
                    "resource_id": 3,
                    "bounding_box": 2,
                    "orientations": [
                        {"cells": [(0, 0), (0, 1), (1, 0), (1, 1)]},  # 2x2 square
                        {"cells": [(0, 0), (1, 0)]},  # 2x1 vertical
                        {"cells": [(0, 0), (0, 1)]}   # 1x2 horizontal
                    ]
                }
            elif resource_id == 4:
                resource = {
                    "resource_id": 4,
                    "bounding_box": 3,
                    "orientations": [
                        {"cells": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]},  # Cross shape
                        {"cells": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]}   # Rotated cross
                    ]
                }
            elif resource_id == 6:
                resource = {
                    "resource_id": 6,
                    "bounding_box": 2,
                    "orientations": [
                        {"cells": [(0, 0), (0, 1), (1, 0)]},  # L shape
                        {"cells": [(0, 0), (1, 0), (1, 1)]},  # Rotated L
                        {"cells": [(0, 1), (1, 0), (1, 1)]},  # Another L
                        {"cells": [(0, 0), (0, 1), (1, 1)]}   # Final L
                    ]
                }
            elif resource_id == 9:
                resource = {
                    "resource_id": 9,
                    "bounding_box": 3,
                    "orientations": [
                        {"cells": [(i, j) for i in range(3) for j in range(3)]},  # 3x3 square
                        {"cells": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]},     # L shape
                    ]
                }
            elif resource_id == 10:
                resource = {
                    "resource_id": 10,
                    "bounding_box": 2,
                    "orientations": [
                        {"cells": [(0, 0), (0, 1)]},  # 1x2 horizontal
                        {"cells": [(0, 0), (1, 0)]}   # 2x1 vertical
                    ]
                }
            elif resource_id == 11:
                resource = {
                    "resource_id": 11,
                    "bounding_box": 4,
                    "orientations": [
                        {"cells": [(0, 0), (0, 1), (0, 2), (0, 3)]},  # 1x4 horizontal
                        {"cells": [(0, 0), (1, 0), (2, 0), (3, 0)]}   # 4x1 vertical
                    ]
                }
            else:
                # Generic resource
                size = min(3, max(2, resource_id % 4 + 1))
                resource = {
                    "resource_id": resource_id,
                    "bounding_box": size,
                    "orientations": [
                        {"cells": [(i, j) for i in range(size) for j in range(size)]},
                        {"cells": [(0, 0), (0, 1)] if size >= 2 else [(0, 0)]}
                    ]
                }
                
            sample_resources.append(resource)
            
        return sample_resources
    
    def can_place(self, resource, orientation, top, left):
        """Check if a resource can be placed at the given position"""
        cells = orientation["cells"]
        resource_id = resource["resource_id"]
        
        # Check bounds and occupation
        for dr, dc in cells:
            r, c = top + dr, left + dc
            if not (0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE):
                return False
            if self.occupied[r][c]:
                return False
        
        # Check 1-block gap requirement for same resource
        for dr, dc in cells:
            r, c = top + dr, left + dc
            # Check surrounding 3x3 area for same resource
            for nr in range(max(0, r - 1), min(self.GRID_SIZE, r + 2)):
                for nc in range(max(0, c - 1), min(self.GRID_SIZE, c + 2)):
                    if self.grid[nr][nc] == resource_id:
                        return False
        
        return True
    
    def place_resource(self, resource, orientation, top, left):
        """Place a resource at the given position"""
        resource_id = resource["resource_id"]
        cells_placed = 0
        
        for dr, dc in orientation["cells"]:
            r, c = top + dr, left + dc
            self.grid[r][c] = resource_id
            self.occupied[r][c] = True
            cells_placed += 1
        
        self.resource_counts[resource_id] += cells_placed
        self.resource_placements[resource_id] += 1
        self.total_placed += cells_placed
    
    def get_diversity_priority(self, resource):
        """Calculate diversity priority - higher for less used resources"""
        resource_id = resource["resource_id"]
        placement_count = self.resource_placements[resource_id]
        
        # Strong preference for resources that haven't been placed yet
        if placement_count == 0:
            return 1000
        
        # Decreasing priority as placement count increases
        return max(1, 100 // (placement_count + 1))
    
    def find_valid_positions(self, resource, orientation):
        """Find all valid positions for a resource orientation"""
        valid_positions = []
        bounding_box = resource["bounding_box"]
        
        for top in range(self.GRID_SIZE - bounding_box + 1):
            for left in range(self.GRID_SIZE - bounding_box + 1):
                if self.can_place(resource, orientation, top, left):
                    valid_positions.append((top, left))
        
        return valid_positions
    
    def choose_best_position(self, valid_positions, prefer_spread=True):
        """Choose the best position from valid positions"""
        if not valid_positions:
            return None
        
        if prefer_spread:
            # Prefer positions that spread resources across the grid
            def position_score(pos):
                top, left = pos
                center_distance = abs(top - self.GRID_SIZE//2) + abs(left - self.GRID_SIZE//2)
                return center_distance
            
            # Sort by distance from center (prefer edges)
            scored_positions = [(position_score(pos), pos) for pos in valid_positions]
            scored_positions.sort(reverse=True)  # Prefer higher distances (edges)
            
            # Choose from top 30% of positions
            top_positions = scored_positions[:max(1, len(scored_positions) * 3 // 10)]
            return random.choice(top_positions)[1]
        else:
            return random.choice(valid_positions)
    
    def optimize_placement_with_diversity(self, resources, max_iterations=10000):
        """Optimize placement with strong emphasis on diversity"""
        placements_made = 0
        iterations = 0
        consecutive_failures = 0
        
        while iterations < max_iterations and consecutive_failures < 50:
            iterations += 1
            
            # Sort resources by diversity priority (highest first)
            resources_by_priority = sorted(resources, key=self.get_diversity_priority, reverse=True)
            
            placed_this_round = False
            
            # Try to place each resource type
            for resource in resources_by_priority:
                # Try each orientation
                for orientation in resource["orientations"]:
                    valid_positions = self.find_valid_positions(resource, orientation)
                    
                    if valid_positions:
                        position = self.choose_best_position(valid_positions, prefer_spread=True)
                        if position:
                            top, left = position
                            self.place_resource(resource, orientation, top, left)
                            placements_made += 1
                            placed_this_round = True
                            consecutive_failures = 0
                            break
                
                if placed_this_round:
                    break
            
            if not placed_this_round:
                consecutive_failures += 1
                # Try random placement as fallback
                if not self.try_random_placement(resources):
                    consecutive_failures += 1
                else:
                    placements_made += 1
                    consecutive_failures = 0
        
        return placements_made
    
    def try_random_placement(self, resources):
        """Try random placement as fallback"""
        attempts = 50
        
        # Prioritize resources that haven't been placed much
        weighted_resources = []
        for resource in resources:
            weight = max(1, 10 - self.resource_placements[resource["resource_id"]])
            weighted_resources.extend([resource] * weight)
        
        for _ in range(attempts):
            resource = random.choice(weighted_resources)
            orientation = random.choice(resource["orientations"])
            
            bounding_box = resource["bounding_box"]
            if bounding_box >= self.GRID_SIZE:
                continue
                
            top = random.randint(0, self.GRID_SIZE - bounding_box)
            left = random.randint(0, self.GRID_SIZE - bounding_box)
            
            if self.can_place(resource, orientation, top, left):
                self.place_resource(resource, orientation, top, left)
                return True
        
        return False
    
    def get_diversity_score(self):
        """Calculate diversity score based on unique resources used"""
        unique_resources = len([count for count in self.resource_placements.values() if count > 0])
        return unique_resources
    
    def get_utilization_score(self):
        """Calculate space utilization score"""
        total_cells = self.GRID_SIZE * self.GRID_SIZE
        occupied_cells = sum(1 for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE) if self.occupied[r][c])
        return occupied_cells / total_cells
    
    def print_stats(self):
        """Print placement statistics"""
        print(f"Total cells placed: {self.total_placed}")
        print(f"Unique resources used: {self.get_diversity_score()}")
        print(f"Space utilization: {self.get_utilization_score():.2%}")
        print(f"Resource placements: {dict(self.resource_placements)}")
        print(f"Resource cell counts: {dict(self.resource_counts)}")
    
    def save_solution(self, filename="solution.txt", level=1, allowed_resources=None):
        """Save the solution to a file"""
        if allowed_resources is None:
            allowed_resources = list(self.resource_counts.keys()) + [1]
        
        solution = {
            "level": level,
            "zoo_size": f"{self.GRID_SIZE}x{self.GRID_SIZE}",
            "resources": sorted(allowed_resources),
            "zoo": self.grid
        }
        
        with open(filename, "w") as f:
            json.dump(solution, f, indent=2)
        
        print(f"Solution saved to {filename}")


def main():
    """Main function to run the enhanced zoo placement optimization"""
    # Level 1 configuration
    level_info = {
        "level": 1,
        "zoo_size": "50x50",
        "resources": [1, 3, 4, 6, 9, 10, 11, 14, 15, 20, 21]
    }
    
    # Initialize optimizer
    optimizer = EnhancedZooPlacementOptimizer(50)
    
    # Load base zoo if available
    optimizer.load_base_zoo("1.txt")
    
    # Load resources
    resources = optimizer.load_resources("resources.json", level_info["resources"])
    
    print(f"Loaded {len(resources)} resources for Level {level_info['level']}")
    print("Starting enhanced optimization with diversity focus...")
    
    # Optimize placement with diversity emphasis
    placements = optimizer.optimize_placement_with_diversity(resources, max_iterations=15000)
    
    print(f"\nOptimization complete! Made {placements} placements.")
    optimizer.print_stats()
    
    # Save solution
    optimizer.save_solution("enhanced_solution.txt", level_info["level"], level_info["resources"])


if __name__ == "__main__":
    main()