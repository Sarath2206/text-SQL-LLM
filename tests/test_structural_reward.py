"""
Unit tests for Structural Reward Component

Tests the StructuralReward class functionality including:
- SELECT clause matching
- WHERE clause matching
- JOIN clause matching
- Partial credit for matching components
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extensions.reward_enhanced import StructuralReward


class TestStructuralReward:
    """Test suite for StructuralReward class."""
    
    @pytest.fixture
    def structural_reward(self):
        """Create StructuralReward instance with default weights."""
        return StructuralReward(
            select_weight=0.3,
            where_weight=0.3,
            join_weight=0.2
        )
    
    def test_exact_match_all_clauses(self, structural_reward):
        """Test exact match of all clauses."""
        sql = "SELECT name, salary FROM employees WHERE department = 'engineering'"
        reward = structural_reward.compute(sql, sql)
        # All three clauses match (no JOIN in this case, so both have no JOIN)
        assert reward == 0.6, f"Expected 0.6 (SELECT + WHERE), got {reward}"
    
    def test_select_clause_match(self, structural_reward):
        """Test SELECT clause matching."""
        gen_sql = "SELECT name, salary FROM employees WHERE department = 'engineering'"
        gt_sql = "SELECT name, salary FROM employees WHERE id = 1"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # Only SELECT matches
        assert reward == 0.3, f"Expected 0.3 (SELECT only), got {reward}"
    
    def test_where_clause_match(self, structural_reward):
        """Test WHERE clause matching."""
        gen_sql = "SELECT name FROM employees WHERE department = 'engineering'"
        gt_sql = "SELECT id FROM employees WHERE department = 'engineering'"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # Only WHERE matches
        assert reward == 0.3, f"Expected 0.3 (WHERE only), got {reward}"
    
    def test_no_match(self, structural_reward):
        """Test completely different queries."""
        gen_sql = "SELECT name FROM employees WHERE id = 1"
        gt_sql = "SELECT salary FROM departments WHERE budget > 1000"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # Nothing matches
        assert reward == 0.0, f"Expected 0.0 (no match), got {reward}"
    
    def test_select_order_independent(self, structural_reward):
        """Test that SELECT column order doesn't matter."""
        gen_sql = "SELECT salary, name FROM employees"
        gt_sql = "SELECT name, salary FROM employees"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # SELECT should match (order-independent)
        assert reward >= 0.3, f"Expected at least 0.3 (SELECT match), got {reward}"
    
    def test_join_clause_match(self, structural_reward):
        """Test JOIN clause matching."""
        gen_sql = """
        SELECT e.name, d.name 
        FROM employees e 
        JOIN departments d ON e.department_id = d.id
        """
        gt_sql = """
        SELECT e.name, d.name 
        FROM employees e 
        JOIN departments d ON e.department_id = d.id
        """
        reward = structural_reward.compute(gen_sql, gt_sql)
        # All clauses match
        assert reward == 0.8, f"Expected 0.8 (SELECT + WHERE + JOIN), got {reward}"
    
    def test_empty_sql(self, structural_reward):
        """Test handling of empty SQL."""
        reward = structural_reward.compute("", "SELECT * FROM employees")
        assert reward == 0.0, "Empty SQL should return 0 reward"
    
    def test_malformed_sql(self, structural_reward):
        """Test handling of malformed SQL."""
        gen_sql = "SELECT FROM WHERE"
        gt_sql = "SELECT name FROM employees"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # Should not crash
        assert isinstance(reward, float), "Should return float even for malformed SQL"
    
    def test_case_insensitive_matching(self, structural_reward):
        """Test that matching is case-insensitive."""
        gen_sql = "SELECT NAME, SALARY FROM employees"
        gt_sql = "SELECT name, salary FROM employees"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # SELECT should match (case-insensitive)
        assert reward >= 0.3, f"Expected at least 0.3 (SELECT match), got {reward}"
    
    def test_custom_weights(self):
        """Test custom weights for each clause."""
        custom_reward = StructuralReward(
            select_weight=0.5,
            where_weight=0.4,
            join_weight=0.3
        )
        sql = "SELECT name FROM employees WHERE id = 1"
        reward = custom_reward.compute(sql, sql)
        # SELECT + WHERE match (no JOIN)
        assert reward == 0.9, f"Expected 0.9 (custom weights), got {reward}"
    
    def test_wildcard_select(self, structural_reward):
        """Test SELECT * matching."""
        gen_sql = "SELECT * FROM employees"
        gt_sql = "SELECT * FROM employees"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # SELECT should match
        assert reward >= 0.3, f"Expected at least 0.3 (SELECT match), got {reward}"
    
    def test_aggregate_functions(self, structural_reward):
        """Test aggregate functions in SELECT."""
        gen_sql = "SELECT COUNT(*), AVG(salary) FROM employees"
        gt_sql = "SELECT COUNT(*), AVG(salary) FROM employees"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # SELECT should match
        assert reward >= 0.3, f"Expected at least 0.3 (SELECT match), got {reward}"
    
    def test_where_with_multiple_predicates(self, structural_reward):
        """Test WHERE clause with multiple predicates."""
        gen_sql = "SELECT name FROM employees WHERE department = 'eng' AND salary > 50000"
        gt_sql = "SELECT name FROM employees WHERE salary > 50000 AND department = 'eng'"
        reward = structural_reward.compute(gen_sql, gt_sql)
        # Both SELECT and WHERE should match (order-independent predicates)
        assert reward >= 0.6, f"Expected at least 0.6 (SELECT + WHERE), got {reward}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
