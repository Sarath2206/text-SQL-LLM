"""
Unit tests for Schema-Aware Reward Component

Tests the SchemaAwareReward class functionality including:
- Detection of non-existent tables
- Detection of non-existent columns
- Penalty calculation
- Valid schema references (no penalty)
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extensions.reward_enhanced import SchemaAwareReward


class TestSchemaAwareReward:
    """Test suite for SchemaAwareReward class."""
    
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
                },
                'projects': {
                    'columns': ['id', 'name', 'start_date', 'end_date']
                }
            }
        }
    
    @pytest.fixture
    def schema_reward(self):
        """Create SchemaAwareReward instance with default weight."""
        return SchemaAwareReward(weight=-0.5)
    
    def test_valid_query_no_penalty(self, schema_reward, sample_schema):
        """Test that valid SQL with correct schema gets no penalty."""
        sql = "SELECT name, salary FROM employees WHERE department = 'engineering'"
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == 0.0, "Valid query should have no penalty"
    
    def test_hallucinated_table_penalty(self, schema_reward, sample_schema):
        """Test detection of non-existent table."""
        sql = "SELECT name FROM customers WHERE id = 1"  # 'customers' doesn't exist
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == -0.5, f"Should penalize 1 hallucinated table, got {reward}"
    
    def test_hallucinated_column_penalty(self, schema_reward, sample_schema):
        """Test detection of non-existent column."""
        sql = "SELECT name, age FROM employees"  # 'age' doesn't exist
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == -0.5, f"Should penalize 1 hallucinated column, got {reward}"
    
    def test_multiple_hallucinations(self, schema_reward, sample_schema):
        """Test detection of multiple hallucinations."""
        # 'customers' table doesn't exist, 'age' and 'email' columns don't exist
        sql = "SELECT name, age, email FROM customers"
        reward = schema_reward.compute(sql, sample_schema)
        # 1 table + 2 columns = 3 hallucinations
        assert reward == -1.5, f"Should penalize 3 hallucinations, got {reward}"
    
    def test_join_with_valid_tables(self, schema_reward, sample_schema):
        """Test JOIN query with valid tables."""
        sql = """
        SELECT e.name, d.name 
        FROM employees e 
        JOIN departments d ON e.department = d.id
        """
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == 0.0, "Valid JOIN should have no penalty"
    
    def test_join_with_invalid_table(self, schema_reward, sample_schema):
        """Test JOIN query with invalid table."""
        sql = """
        SELECT e.name, c.name 
        FROM employees e 
        JOIN customers c ON e.id = c.employee_id
        """
        reward = schema_reward.compute(sql, sample_schema)
        assert reward < 0, "Should penalize hallucinated table in JOIN"
    
    def test_case_insensitive_matching(self, schema_reward, sample_schema):
        """Test that schema matching is case-insensitive."""
        sql = "SELECT NAME, SALARY FROM EMPLOYEES"  # Uppercase
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == 0.0, "Case-insensitive matching should work"
    
    def test_empty_sql(self, schema_reward, sample_schema):
        """Test handling of empty SQL."""
        reward = schema_reward.compute("", sample_schema)
        assert reward == 0.0, "Empty SQL should return neutral reward"
    
    def test_empty_schema(self, schema_reward):
        """Test handling of empty schema."""
        sql = "SELECT * FROM employees"
        reward = schema_reward.compute(sql, {})
        assert reward == 0.0, "Empty schema should return neutral reward"
    
    def test_malformed_sql(self, schema_reward, sample_schema):
        """Test handling of malformed SQL."""
        sql = "SELECT FROM WHERE"  # Invalid SQL
        reward = schema_reward.compute(sql, sample_schema)
        # Should not crash, return neutral reward
        assert isinstance(reward, float), "Should return float even for malformed SQL"
    
    def test_custom_weight(self, sample_schema):
        """Test custom penalty weight."""
        custom_reward = SchemaAwareReward(weight=-1.0)
        sql = "SELECT name FROM customers"  # 1 hallucinated table
        reward = custom_reward.compute(sql, sample_schema)
        assert reward == -1.0, "Should use custom weight"
    
    def test_wildcard_select(self, schema_reward, sample_schema):
        """Test SELECT * doesn't cause false positives."""
        sql = "SELECT * FROM employees"
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == 0.0, "SELECT * should not be penalized"
    
    def test_aggregate_functions(self, schema_reward, sample_schema):
        """Test aggregate functions with valid columns."""
        sql = "SELECT COUNT(*), AVG(salary) FROM employees"
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == 0.0, "Aggregate functions with valid columns should work"
    
    def test_subquery(self, schema_reward, sample_schema):
        """Test subquery with valid schema."""
        sql = """
        SELECT name FROM employees 
        WHERE department IN (SELECT id FROM departments WHERE budget > 100000)
        """
        reward = schema_reward.compute(sql, sample_schema)
        assert reward == 0.0, "Valid subquery should have no penalty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
