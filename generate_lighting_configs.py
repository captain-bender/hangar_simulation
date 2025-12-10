#!/usr/bin/env python3
"""
Standalone Lighting Configuration Generator for Synthetic Dataset

Generates varied lighting configurations independently from rover positions.
Each configuration is assigned an ID (position_001, position_002, etc.) that
will later be matched with rover positions during rendering.

Configuration Parameters:
- Power: 400 ± 150 (range: 250-550)
- Rotation: FIXED (no variation)
- Location X: FIXED
- Location Y: ±5m variation
- Location Z: ±2m variation
"""

import json
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LightConfig:
    """Configuration for a single light source"""
    name: str
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    power: float


@dataclass
class SceneLightingConfig:
    """Complete lighting configuration for a single scene"""
    position_id: str
    lights: List[LightConfig]


class LightingConfigGenerator:
    """Generate varied lighting configurations"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator with reproducible seed
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
        # Base configurations for the two dominant lights
        self.lights_base = [
            {
                'name': 'Right Side Big',
                'location': [-38.375, 10.224, 15.385],
                'rotation': [-75, 0, 90],  # Fixed rotation
                'power': 400
            },
            {
                'name': 'Left Side Big',
                'location': [38.374, 10.224, 15.384],
                'rotation': [75, 0, 90],  # Fixed rotation
                'power': 400
            }
        ]
        
        # Variation ranges
        self.power_variation = 150  # ±150 around 400
        self.y_variation = 5.0      # ±5m
        self.z_variation = 2.0      # ±2m
    
    def generate_light_config(self, base_light: dict) -> LightConfig:
        """
        Generate a single light configuration with random variations
        
        Args:
            base_light: Base light configuration dict
            
        Returns:
            LightConfig with randomized parameters
        """
        # Power variation: 400 ± 150 (250 to 550)
        power = base_light['power'] + random.uniform(
            -self.power_variation, 
            self.power_variation
        )
        
        # Location variation
        # X: Fixed (no variation)
        # Y: ±5m
        # Z: ±2m
        x = base_light['location'][0]  # Fixed X
        y = base_light['location'][1] + random.uniform(-self.y_variation, self.y_variation)
        z = base_light['location'][2] + random.uniform(-self.z_variation, self.z_variation)
        
        # Rotation: Fixed (no variation)
        rotation = tuple(base_light['rotation'])
        
        return LightConfig(
            name=base_light['name'],
            location=(x, y, z),
            rotation=rotation,
            power=power
        )
    
    def generate_scene_config(self, position_id: str) -> SceneLightingConfig:
        """
        Generate complete lighting configuration for one position
        
        Args:
            position_id: Identifier for the position (e.g., 'position_001')
            
        Returns:
            SceneLightingConfig with all lights configured
        """
        lights = [self.generate_light_config(base) for base in self.lights_base]
        
        return SceneLightingConfig(
            position_id=position_id,
            lights=lights
        )
    
    def generate_configs(self, num_configs: int, start_id: int = 1) -> List[SceneLightingConfig]:
        """
        Generate specified number of lighting configurations
        
        Args:
            num_configs: Number of configurations to generate
            start_id: Starting ID number (default: 1)
            
        Returns:
            List of SceneLightingConfig objects
        """
        configs = []
        
        for i in range(num_configs):
            position_id = f"position_{start_id + i:03d}"
            config = self.generate_scene_config(position_id)
            configs.append(config)
        
        return configs
    
    def configs_to_dict(self, configs: List[SceneLightingConfig]) -> dict:
        """
        Convert lighting configurations to dictionary format for JSON export
        
        Args:
            configs: List of SceneLightingConfig objects
            
        Returns:
            Dictionary ready for JSON serialization
        """
        return {
            'lighting_configs': [
                {
                    'position_id': config.position_id,
                    'lights': [
                        {
                            'name': light.name,
                            'location': list(light.location),
                            'rotation': list(light.rotation),
                            'power': round(light.power, 2)
                        }
                        for light in config.lights
                    ]
                }
                for config in configs
            ]
        }
    
    def save_to_json(self, configs: List[SceneLightingConfig], filepath: str):
        """
        Save lighting configurations to JSON file
        
        Args:
            configs: List of SceneLightingConfig objects
            filepath: Output file path
        """
        data = self.configs_to_dict(configs)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Saved {len(configs)} lighting configurations to {filepath}")
    
    def print_summary(self, configs: List[SceneLightingConfig]):
        """Print summary statistics of generated configurations"""
        all_powers = []
        all_y_offsets = []
        all_z_offsets = []
        
        for config in configs:
            for light in config.lights:
                all_powers.append(light.power)
                
                # Calculate offsets from base
                base = next(l for l in self.lights_base if l['name'] == light.name)
                y_offset = light.location[1] - base['location'][1]
                z_offset = light.location[2] - base['location'][2]
                
                all_y_offsets.append(y_offset)
                all_z_offsets.append(z_offset)
        
        print("\n" + "="*70)
        print("LIGHTING CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Total configurations: {len(configs)}")
        print(f"Lights per configuration: {len(configs[0].lights)}")
        print(f"ID range: {configs[0].position_id} to {configs[-1].position_id}")
        print(f"\nPower statistics:")
        print(f"  Range: {min(all_powers):.1f} - {max(all_powers):.1f}")
        print(f"  Mean: {sum(all_powers)/len(all_powers):.1f}")
        print(f"\nY-offset statistics (meters):")
        print(f"  Range: {min(all_y_offsets):.2f} - {max(all_y_offsets):.2f}")
        print(f"  Mean: {sum(all_y_offsets)/len(all_y_offsets):.2f}")
        print(f"\nZ-offset statistics (meters):")
        print(f"  Range: {min(all_z_offsets):.2f} - {max(all_z_offsets):.2f}")
        print(f"  Mean: {sum(all_z_offsets)/len(all_z_offsets):.2f}")
        print("="*70 + "\n")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate varied lighting configurations for synthetic dataset'
    )
    parser.add_argument(
        '--num-configs',
        type=int,
        required=True,
        help='Number of lighting configurations to generate'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for lighting configuration JSON'
    )
    parser.add_argument(
        '--start-id',
        type=int,
        default=1,
        help='Starting ID number (default: 1 for position_001)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--power-variation',
        type=float,
        default=150,
        help='Power variation range in watts (default: ±150)'
    )
    parser.add_argument(
        '--y-variation',
        type=float,
        default=5.0,
        help='Y-axis variation in meters (default: ±5.0)'
    )
    parser.add_argument(
        '--z-variation',
        type=float,
        default=2.0,
        help='Z-axis variation in meters (default: ±2.0)'
    )
    
    args = parser.parse_args()
    
    # Create generator with custom parameters
    print(f"[INFO] Initializing generator with seed {args.seed}")
    generator = LightingConfigGenerator(seed=args.seed)
    generator.power_variation = args.power_variation
    generator.y_variation = args.y_variation
    generator.z_variation = args.z_variation
    
    # Generate configurations
    print(f"[INFO] Generating {args.num_configs} lighting configurations...")
    configs = generator.generate_configs(args.num_configs, args.start_id)
    
    # Print summary
    generator.print_summary(configs)
    
    # Save to JSON
    generator.save_to_json(configs, args.output)
    
    print(f"[SUCCESS] Generation complete!")


if __name__ == '__main__':
    main()
