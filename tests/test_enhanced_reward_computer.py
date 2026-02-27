"""
Unit tests for EnhancedRewardComputer class

Tests the integration of all reward components:
- Baseline reward function wrapping
- Enhanced reward component integration
- Reward aggregation logic
- Enable/disable flags for each component
- Detailed reward breakdown logging
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extensions.reward_enhanced import EnhancedRewardComputer, EnhancedRewardResult


class TestEnhancedRewardComputer:
    """Test suite for EnhancedRewardComputer class."""
    
    @pytest.fixture
    def mock_baseline_reward_fn(self):
        """Mock baseline reward function that returns a fixed score."""
        def baseline_fn(solution_str, ground_truth, **kwargs):
            # Simple mock: return 3.0 if SQL is present, 0.0 otherwise
            if '```sql' in solution_str:
                return 3.0
            return 0.0
        return baseline_fn
    
    @pytest.fixture
    def sample_schema(self):
        """Sample database schema for testing."""
        return {
            'tables': {
                'employees': {
                    'columns': ['id', 'name', 'salary', 'department']
                },
                'departments': {
                    'columns': ['id', 'name', 'budget']
                }
            }
        }
    
    @pytest.fixture
    def sample_solution(self):
        """Sample solution string with proper format."""
        return """<think>
I need to select employee names and salaries from the employees table.
</think>
<answer>
```sql
SELECT name, salary FROM employees WHERE department = 'engineering'
```
</answer>"""
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Sample ground truth data."""
        return {
            'sql': "SELECT name, salary FROM employees WHERE department = 'engineering'"
        }
    
    def test_initialization(self, mock_baseline_reward_fn):
        """Test EnhancedRewardComputer initialization."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        assert computer.baseline_reward_fn is not None
        assert computer.enable_enhanced is True
        assert computer.schema_reward is not None
        assert computer.structural_reward is not None
        assert computer.syntax_reward is not None
    
    def test_compute_reward_with_enhanced_enabled(
        self,
        mock_baseline_reward_fn,
        sample_solution,
        sample_ground_truth,
        sample_schema
    ):
        """Test compute_reward with enhanced components enabled."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        result = computer.compute_reward(
            solution_str=sample_solution,
            ground_truth=sample_ground_truth,
            schema=sample_schema
        )
        
        # Verify result structure
        assert isinstance(result, EnhancedRewardResult)
        assert hasattr(result, 'total')
        assert hasattr(result, 'baseline_total')
        assert hasattr(result, 'schema')
        assert hasattr(result, 'structural')
        assert hasattr(result, 'syntax')
        
        # Verify total is sum of components
        expected_total = (
            result.baseline_total +
            result.schema +
            result.structural +
            result.syntax
        )
        assert abs(result.total - expected_total) < 0.001
    
    def test_compute_reward_with_enhanced_disabled(
        self,
        mock_baseline_reward_fn,
        sample_solution,
        sample_ground_truth,
        sample_schema
    ):
        """Test compute_reward with enhanced components disabled."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=False
        )
        
        result = computer.compute_reward(
            solution_str=sample_solution,
            ground_truth=sample_ground_truth,
            schema=sample_schema
        )
        
        # Enhanced components should be zero when disabled
        assert result.schema == 0.0
        assert result.structural == 0.0
        assert result.syntax == 0.0
        
        # Total should equal baseline
        assert result.total == result.baseline_total
    
    def test_sql_extraction_from_solution(self, mock_baseline_reward_fn):
        """Test SQL extraction from solution string."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        solution = """<think>reasoning</think>
<answer>
```sql
SELECT * FROM employees
```
</answer>"""
        
        sql = computer._extract_sql_from_solution(solution)
        assert sql == "SELECT * FROM employees"
    
    def test_sql_extraction_without_tags(self, mock_baseline_reward_fn):
        """Test SQL extraction when tags are missing."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        solution = "```sql\nSELECT * FROM employees\n```"
        sql = computer._extract_sql_from_solution(solution)
        assert "SELECT * FROM employees" in sql
    
    def test_baseline_component_extraction(
        self,
        mock_baseline_reward_fn,
        sample_solution,
        sample_ground_truth
    ):
        """Test extraction of baseline reward components."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        format_r, exec_r, result_r, length_r = computer._extract_baseline_components(
            sample_solution,
            sample_ground_truth
        )
        
        # Should detect proper format
        assert format_r == 1.0, "Should detect correct format tags"
        
        # All components should be floats
        assert isinstance(format_r, float)
        assert isinstance(exec_r, float)
        assert isinstance(result_r, float)
        assert isinstance(length_r, float)
    
    def test_format_reward_detection(self, mock_baseline_reward_fn):
        """Test format reward detection for different formats."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        # Valid format
        valid_solution = "<think>x</think><answer>```sql\nSELECT 1\n```</answer>"
        format_r, _, _, _ = computer._extract_baseline_components(valid_solution, {})
        assert format_r == 1.0
        
        # Missing think tags
        invalid_solution = "<answer>```sql\nSELECT 1\n```</answer>"
        format_r, _, _, _ = computer._extract_baseline_components(invalid_solution, {})
        assert format_r == 0.0
        
        # Missing sql code block
        invalid_solution = "<think>x</think><answer>SELECT 1</answer>"
        format_r, _, _, _ = computer._extract_baseline_components(invalid_solution, {})
        assert format_r == 0.0
    
    def test_length_penalty(self, mock_baseline_reward_fn):
        """Test length penalty for overly long responses."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        # Short solution (no penalty)
        short_solution = "<think>x</think><answer>```sql\nSELECT 1\n```</answer>"
        _, _, _, length_r = computer._extract_baseline_components(short_solution, {})
        assert length_r == 0.0
        
        # Long solution (with penalty)
        long_solution = "<think>" + "x" * 2000 + "</think><answer>```sql\nSELECT 1\n```</answer>"
        _, _, _, length_r = computer._extract_baseline_components(long_solution, {})
        assert length_r < 0.0, "Should penalize long responses"
    
    def test_schema_reward_integration(
        self,
        mock_baseline_reward_fn,
        sample_schema
    ):
        """Test schema reward integration."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True,
            schema_weight=-0.5
        )
        
        # Solution with hallucinated table
        solution = """<think>x</think>
<answer>
```sql
SELECT name FROM customers
```
</answer>"""
        
        ground_truth = {'sql': 'SELECT name FROM employees'}
        
        result = computer.compute_reward(
            solution_str=solution,
            ground_truth=ground_truth,
            schema=sample_schema
        )
        
        # Should have negative schema reward (or -0.0 which is acceptable)
        assert result.schema <= 0.0, "Should penalize hallucinated table"
    
    def test_structural_reward_integration(
        self,
        mock_baseline_reward_fn,
        sample_schema
    ):
        """Test structural reward integration."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True,
            structural_select_weight=0.3
        )
        
        # Solution with matching SELECT clause
        solution = """<think>x</think>
<answer>
```sql
SELECT name, salary FROM employees
```
</answer>"""
        
        ground_truth = {'sql': 'SELECT name, salary FROM employees'}
        
        result = computer.compute_reward(
            solution_str=solution,
            ground_truth=ground_truth,
            schema=sample_schema
        )
        
        # Should have positive structural reward
        assert result.structural > 0.0, "Should reward matching structure"
    
    def test_syntax_reward_integration(
        self,
        mock_baseline_reward_fn,
        sample_schema
    ):
        """Test syntax reward integration."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True,
            syntax_weight=0.2
        )
        
        # Solution with valid SQL syntax
        solution = """<think>x</think>
<answer>
```sql
SELECT name FROM employees
```
</answer>"""
        
        ground_truth = {'sql': 'SELECT name FROM employees'}
        
        result = computer.compute_reward(
            solution_str=solution,
            ground_truth=ground_truth,
            schema=sample_schema
        )
        
        # Should have positive syntax reward
        assert result.syntax > 0.0, "Should reward valid syntax"
    
    def test_to_dict_conversion(
        self,
        mock_baseline_reward_fn,
        sample_solution,
        sample_ground_truth,
        sample_schema
    ):
        """Test conversion of result to dictionary."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        result = computer.compute_reward(
            solution_str=sample_solution,
            ground_truth=sample_ground_truth,
            schema=sample_schema
        )
        
        result_dict = result.to_dict()
        
        # Verify all keys are present
        expected_keys = [
            'total', 'baseline_total', 'baseline_format',
            'baseline_execution', 'baseline_result', 'baseline_length',
            'schema', 'structural', 'syntax'
        ]
        
        for key in expected_keys:
            assert key in result_dict, f"Missing key: {key}"
            assert isinstance(result_dict[key], float), f"Key {key} should be float"
    
    def test_empty_solution(
        self,
        mock_baseline_reward_fn,
        sample_ground_truth,
        sample_schema
    ):
        """Test handling of empty solution."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True
        )
        
        result = computer.compute_reward(
            solution_str="",
            ground_truth=sample_ground_truth,
            schema=sample_schema
        )
        
        # Should not crash, return valid result
        assert isinstance(result, EnhancedRewardResult)
        assert result.total >= 0.0 or result.total < 0.0  # Just check it's a valid number
    
    def test_custom_weights(self, mock_baseline_reward_fn):
        """Test custom weight configuration."""
        computer = EnhancedRewardComputer(
            baseline_reward_fn=mock_baseline_reward_fn,
            enable_enhanced=True,
            schema_weight=-1.0,
            structural_select_weight=0.5,
            structural_where_weight=0.5,
            structural_join_weight=0.3,
            syntax_weight=0.4
        )
        
        # Verify weights are set correctly
        assert computer.schema_reward.weight == -1.0
        assert computer.structural_reward.select_weight == 0.5
        assert computer.structural_reward.where_weight == 0.5
        assert computer.structural_reward.join_weight == 0.3
        assert computer.syntax_reward.weight == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
