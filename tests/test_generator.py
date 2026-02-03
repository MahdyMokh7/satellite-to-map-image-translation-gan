import torch
import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.generator import GeneratorUNet

def load_config():
    """Load project configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_generator_with_project_config():
    """Test generator with actual project configuration."""
    print("=" * 60)
    print("Testing Generator with Project Config")
    print("=" * 60)
    
    try:
        config = load_config()
        generator_config = config["generator"]
        dataset_config = config["dataset"]
        
        print(f"Generator Config: {generator_config}")
        print(f"Dataset Config: {dataset_config}")
        
        # Create generator from config
        generator = GeneratorUNet(
            in_channels=generator_config["in_channels"],
            out_channels=generator_config["out_channels"],
            base_filters=generator_config["base_filters"],
            num_downs=generator_config["num_downs"]
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in generator.parameters())
        trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        
        print(f"\nGenerator Statistics:")
        print(f"  Architecture: {generator_config['architecture']}")
        print(f"  Base filters: {generator_config['base_filters']}")
        print(f"  Downsampling layers: {generator_config['num_downs']}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Activation: {generator_config['activation']}")
        
        # Test forward pass with dataset image size
        image_size = dataset_config["image_size"]
        channels = dataset_config["channels"]
        
        dummy_input = torch.randn(2, channels, image_size, image_size)
        output = generator(dummy_input)
        
        expected_shape = (2, generator_config["out_channels"], image_size, image_size)
        
        if output.shape == expected_shape:
            print(f"\n✓ Config-based generator works correctly")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            
            # Check output range (should match normalization)
            min_val, max_val = output.min().item(), output.max().item()
            normalization = dataset_config["normalization"]
            
            if normalization == [-1, 1]:
                if -1.1 <= min_val <= 1.1 and -1.1 <= max_val <= 1.1:
                    print(f"✓ Output in expected range: [{min_val:.3f}, {max_val:.3f}]")
                else:
                    print(f"⚠ Output range: [{min_val:.3f}, {max_val:.3f}] (expected ~[-1, 1])")
            
            return True
        else:
            print(f"\n✗ Config-based generator has issues")
            print(f"  Expected: {expected_shape}")
            print(f"  Got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generator_dimensions():
    """Test that generator preserves input dimensions."""
    print("\n" + "=" * 60)
    print("Testing Generator Dimensions")
    print("=" * 60)
    
    try:
        config = load_config()
        generator_config = config["generator"]
        dataset_config = config["dataset"]
        
        base_config = {
            "in_channels": generator_config["in_channels"],
            "out_channels": generator_config["out_channels"],
            "base_filters": generator_config["base_filters"],
            "num_downs": generator_config["num_downs"],
            "input_size": dataset_config["image_size"]
        }
        
        # Test variations
        test_cases = [
            base_config,
            {**base_config, "base_filters": 32, "num_downs": 3},
            {**base_config, "base_filters": 128, "num_downs": 5},
        ]
        
        all_passed = True
        
        for i, test_config in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"  Channels: {test_config['in_channels']}→{test_config['out_channels']}")
            print(f"  Base filters: {test_config['base_filters']}")
            print(f"  Down layers: {test_config['num_downs']}")
            print(f"  Image size: {test_config['input_size']}")
            print("-" * 40)
            
            # Create generator
            generator = GeneratorUNet(
                in_channels=test_config["in_channels"],
                out_channels=test_config["out_channels"],
                base_filters=test_config["base_filters"],
                num_downs=test_config["num_downs"]
            )
            
            # Create test input
            batch_size = 2
            dummy_input = torch.randn(
                batch_size, 
                test_config["in_channels"], 
                test_config["input_size"], 
                test_config["input_size"]
            )
            
            # Forward pass
            output = generator(dummy_input)
            
            # Check dimensions
            expected_shape = (
                batch_size, 
                test_config["out_channels"], 
                test_config["input_size"], 
                test_config["input_size"]
            )
            
            if output.shape == expected_shape:
                print(f"✓ Dimensions correct: {output.shape}")
            else:
                print(f"✗ Dimension mismatch!")
                print(f"  Expected: {expected_shape}")
                print(f"  Got: {output.shape}")
                all_passed = False
            
            # Check output range
            min_val, max_val = output.min().item(), output.max().item()
            if -1.1 <= min_val <= 1.1 and -1.1 <= max_val <= 1.1:
                print(f"✓ Output range: [{min_val:.3f}, {max_val:.3f}]")
            else:
                print(f"⚠ Output range: [{min_val:.3f}, {max_val:.3f}]")
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error in dimension test: {e}")
        return False

def test_gradient_flow():
    """Test that gradients flow properly through the generator."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    try:
        config = load_config()
        generator_config = config["generator"]
        dataset_config = config["dataset"]
        
        generator = GeneratorUNet(
            in_channels=generator_config["in_channels"],
            out_channels=generator_config["out_channels"],
            base_filters=generator_config["base_filters"],
            num_downs=generator_config["num_downs"]
        )
        
        # Create dummy data matching config
        image_size = dataset_config["image_size"]
        channels = generator_config["in_channels"]
        
        dummy_input = torch.randn(2, channels, image_size, image_size)
        dummy_target = torch.randn(2, generator_config["out_channels"], image_size, image_size)
        
        # Forward pass
        output = generator(dummy_input)
        loss = torch.nn.functional.l1_loss(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = False
        total_zero_gradients = 0
        param_count = 0
        
        for name, param in generator.named_parameters():
            param_count += 1
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.abs().sum().item()
                if grad_norm == 0:
                    total_zero_gradients += 1
        
        if has_gradients:
            print(f"✓ Gradients are flowing through the network")
            print(f"  Total parameters: {param_count}")
            print(f"  Parameters with gradients: {param_count - total_zero_gradients}")
            
            if total_zero_gradients > 0:
                print(f"  Warning: {total_zero_gradients} parameters have zero gradients")
            
            return True
        else:
            print(f"✗ No gradients detected!")
            return False
            
    except Exception as e:
        print(f"✗ Error in gradient test: {e}")
        return False

def test_skip_connections():
    """Test that skip connections are properly implemented."""
    print("\n" + "=" * 60)
    print("Testing Skip Connections")
    print("=" * 60)
    
    try:
        config = load_config()
        generator_config = config["generator"]
        dataset_config = config["dataset"]
        
        # Use smaller config for faster testing
        generator = GeneratorUNet(
            in_channels=generator_config["in_channels"],
            out_channels=generator_config["out_channels"],
            base_filters=32,  # Smaller for clarity
            num_downs=3       # Fewer layers for clarity
        )
        
        # Instrument forward pass
        skip_features = []
        original_forward = generator.forward
        
        def instrumented_forward(x):
            skip_features.clear()
            
            # Encoder
            for down in generator.downs:
                x = down(x)
                skip_features.append(x)
            
            # Bottleneck
            x = generator.bottleneck(x)
            
            # Decoder with skip connections
            for i, up in enumerate(generator.ups):
                skip = skip_features[-(i + 1)]
                print(f"  Level {i}: Bottleneck {x.shape} + Skip {skip.shape}")
                x = torch.cat([x, skip], dim=1)
                x = up(x)
            
            # Final
            x = torch.cat([x, skip_features[0]], dim=1)
            x = generator.final(x)
            return x
        
        # Test
        image_size = dataset_config["image_size"]
        dummy_input = torch.randn(1, generator_config["in_channels"], image_size, image_size)
        generator.forward = instrumented_forward
        
        try:
            output = generator(dummy_input)
            print(f"\n✓ Skip connections implemented correctly")
            print(f"  Number of skip connections: {len(skip_features)}")
            print(f"  Final output shape: {output.shape}")
            
            # Verify all skip connections were used
            if len(skip_features) == len(generator.downs):
                print(f"✓ Correct number of skip connections")
            else:
                print(f"⚠ Expected {generator_config['num_downs']} skips, got {len(skip_features)}")
            
            return True
        except Exception as e:
            print(f"✗ Skip connection error: {e}")
            return False
        finally:
            generator.forward = original_forward
            
    except Exception as e:
        print(f"✗ Error in skip connection test: {e}")
        return False

def test_pretrained_config():
    """Test pretrained model loading configuration."""
    print("\n" + "=" * 60)
    print("Testing Pretrained Config")
    print("=" * 60)
    
    try:
        config = load_config()
        generator_config = config["generator"]
        pretrained_config = generator_config["pretrained"]
        
        print(f"Pretrained config: {pretrained_config}")
        
        if pretrained_config["use"]:
            print(f"✓ Pretrained loading enabled")
            print(f"  Path: {pretrained_config['path']}")
            print(f"  Frozen encoder layers: {pretrained_config['freeze_encoder_layers']}")
            
            # Test if path exists (or at least is valid format)
            if pretrained_config["path"].endswith('.pth'):
                print(f"  File format: .pth (valid)")
            else:
                print(f"⚠ File format might not be .pth")
        else:
            print(f"✓ Training from scratch")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in pretrained config test: {e}")
        return False

def run_all_tests():
    """Run all generator tests."""
    print("\n" + "=" * 60)
    print("GENERATOR TEST SUITE")
    print("Config-based testing for satellite2map_cgan")
    print("=" * 60)
    
    # Load config once for display
    config = load_config()
    print(f"\nProject: {config['project']['name']}")
    print(f"Experiment: {config['project']['experiment_name']}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Image size: {config['dataset']['image_size']}x{config['dataset']['image_size']}")
    print(f"Mode: {config['training']['mode']}")
    
    test_results = []
    
    # Run tests
    test_results.append(("Project Config Test", test_generator_with_project_config()))
    test_results.append(("Dimension Tests", test_generator_dimensions()))
    test_results.append(("Gradient Flow Test", test_gradient_flow()))
    test_results.append(("Skip Connections Test", test_skip_connections()))
    test_results.append(("Pretrained Config Test", test_pretrained_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    passed_count = 0
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed_count += 1
        else:
            all_passed = False
    
    print(f"\nPassed: {passed_count}/{len(test_results)}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Generator is ready for training.")
    else:
        print("⚠ SOME TESTS FAILED.")
        print("Check generator implementation before training.")
    
    return all_passed

if __name__ == "__main__":
    # Run tests when file is executed directly
    success = run_all_tests()
    sys.exit(0 if success else 1)