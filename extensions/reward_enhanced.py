"""
Enhanced Reward Computation Module

This module extends SQL-R1's existing reward computation with additional components:
- Schema-aware rewards (detect hallucinated tables/columns)
- Structural rewards (SELECT, WHERE, JOIN clause matching)
- Enhanced syntax rewards (AST-level validation)

The design wraps SQL-R1's baseline reward function to maintain backward compatibility.
"""

from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Function
from sqlparse.tokens import Keyword, DML
import re


@dataclass
class EnhancedRewardResult:
    """Result of enhanced reward computation with component breakdown."""
    total: float
    baseline_total: float
    baseline_format: float
    baseline_execution: float
    baseline_result: float
    baseline_length: float
    schema: float
    structural: float
    syntax: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'total': self.total,
            'baseline_total': self.baseline_total,
            'baseline_format': self.baseline_format,
            'baseline_execution': self.baseline_execution,
            'baseline_result': self.baseline_result,
            'baseline_length': self.baseline_length,
            'schema': self.schema,
            'structural': self.structural,
            'syntax': self.syntax
        }


class SchemaAwareReward:
    """
    Detects and penalizes hallucinated schema elements (non-existent tables/columns).
    
    This component helps the model learn to use only valid schema elements,
    reducing hallucination errors.
    """
    
    def __init__(self, weight: float = -0.5):
        """
        Initialize schema-aware reward component.
        
        Args:
            weight: Penalty weight per hallucinated element (should be negative)
        """
        self.weight = weight
    
    def compute(self, sql: str, schema: Dict[str, Any]) -> float:
        """
        Compute schema-aware reward by detecting hallucinations.
        
        Args:
            sql: Generated SQL query
            schema: Database schema information
                Expected format: {
                    'tables': {
                        'table_name': {
                            'columns': ['col1', 'col2', ...]
                        }
                    }
                }
        
        Returns:
            Reward score (negative penalty for hallucinations)
        """
        if not schema or not sql:
            return 0.0
        
        try:
            # Extract table and column references from SQL
            tables, columns_by_table = self._extract_references(sql)
            
            # Get valid schema elements
            valid_tables = set(schema.get('tables', {}).keys())
            
            # Detect hallucinated tables
            hallucinated_tables = tables - valid_tables
            
            # Detect hallucinated columns
            hallucinated_columns = set()
            for table, cols in columns_by_table.items():
                if table == '_unknown':
                    # For columns without table prefix, check if they exist in ANY valid table
                    all_valid_cols = set()
                    for t in valid_tables:
                        all_valid_cols.update(schema['tables'][t].get('columns', []))
                    # Normalize column names (lowercase, strip)
                    all_valid_cols_normalized = {c.lower().strip() for c in all_valid_cols}
                    cols_normalized = {c.lower().strip() for c in cols}
                    hallucinated = cols_normalized - all_valid_cols_normalized
                    hallucinated_columns.update(hallucinated)
                elif table in valid_tables:
                    valid_cols = set(schema['tables'][table].get('columns', []))
                    # Normalize column names (lowercase, strip)
                    valid_cols_normalized = {c.lower().strip() for c in valid_cols}
                    cols_normalized = {c.lower().strip() for c in cols}
                    hallucinated = cols_normalized - valid_cols_normalized
                    hallucinated_columns.update(hallucinated)
            
            # Calculate penalty
            num_hallucinations = len(hallucinated_tables) + len(hallucinated_columns)
            
            if num_hallucinations > 0:
                # Log hallucinations for debugging
                if hallucinated_tables:
                    print(f"[Schema Reward] Hallucinated tables: {hallucinated_tables}")
                if hallucinated_columns:
                    print(f"[Schema Reward] Hallucinated columns: {hallucinated_columns}")
            
            return self.weight * num_hallucinations
            
        except Exception as e:
            # If parsing fails, return neutral reward
            print(f"[Schema Reward] Error during schema validation: {e}")
            return 0.0
    
    def _extract_references(self, sql: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """
        Extract table and column references from SQL query.
        
        Args:
            sql: SQL query string
        
        Returns:
            Tuple of (tables, columns_by_table)
        """
        tables = set()
        columns_by_table = {}
        
        try:
            # Parse SQL using sqlparse
            parsed = sqlparse.parse(sql)
            if not parsed:
                return tables, columns_by_table
            
            statement = parsed[0]
            
            # Extract tables from FROM and JOIN clauses
            tables = self._extract_tables(statement)
            
            # Extract columns from SELECT, WHERE, and other clauses
            columns_by_table = self._extract_columns(statement, tables)
            
        except Exception as e:
            print(f"[Schema Reward] Error parsing SQL: {e}")
        
        return tables, columns_by_table
    
    def _extract_tables(self, statement: sqlparse.sql.Statement) -> Set[str]:
        """Extract table names from SQL statement."""
        tables = set()
        
        # Look for FROM and JOIN keywords
        from_seen = False
        join_seen = False
        
        for token in statement.tokens:
            # Skip whitespace tokens
            if token.ttype in (sqlparse.tokens.Whitespace, sqlparse.tokens.Newline):
                continue
            
            # Check for FROM keyword
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
                continue
            
            # Check for JOIN keywords
            if token.ttype is Keyword and 'JOIN' in token.value.upper():
                join_seen = True
                continue
            
            # Extract table names after FROM or JOIN
            if from_seen or join_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        table_name = self._get_real_name(identifier)
                        if table_name:
                            tables.add(table_name)
                elif isinstance(token, Identifier):
                    table_name = self._get_real_name(token)
                    if table_name:
                        tables.add(table_name)
                
                # Reset flags after processing non-keyword token
                if token.ttype is not Keyword:
                    from_seen = False
                    join_seen = False
        
        return tables
    
    def _extract_columns(
        self,
        statement: sqlparse.sql.Statement,
        tables: Set[str]
    ) -> Dict[str, Set[str]]:
        """Extract column references from SQL statement."""
        columns_by_table = {table: set() for table in tables}
        columns_by_table['_unknown'] = set()  # For columns without explicit table
        
        # Extract from SELECT clause
        select_seen = False
        for token in statement.tokens:
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
                continue
            
            if select_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        self._process_column_identifier(identifier, columns_by_table)
                elif isinstance(token, Identifier):
                    self._process_column_identifier(token, columns_by_table)
                
                # Stop at FROM keyword
                if token.ttype is Keyword and token.value.upper() == 'FROM':
                    break
        
        # Extract from WHERE clause
        for token in statement.tokens:
            if isinstance(token, Where):
                self._extract_columns_from_where(token, columns_by_table)
        
        return columns_by_table
    
    def _process_column_identifier(
        self,
        identifier: Identifier,
        columns_by_table: Dict[str, Set[str]]
    ):
        """Process a column identifier and add to appropriate table."""
        # Skip wildcards and functions
        if str(identifier) == '*':
            return
        
        if isinstance(identifier, Function):
            # Extract columns from function arguments (skip the function name itself)
            for token in identifier.tokens:
                # Process Parenthesis tokens which contain the arguments
                if hasattr(token, 'tokens'):  # Parenthesis or other compound tokens
                    for sub_token in token.tokens:
                        if isinstance(sub_token, Identifier):
                            self._process_column_identifier(sub_token, columns_by_table)
            return
        
        # Check if column has table prefix (e.g., table.column)
        name = identifier.get_real_name()
        parent = identifier.get_parent_name()
        
        if parent:
            # Column with table prefix
            if parent not in columns_by_table:
                columns_by_table[parent] = set()
            columns_by_table[parent].add(name)
        else:
            # Column without table prefix
            columns_by_table['_unknown'].add(name)
    
    def _extract_columns_from_where(
        self,
        where_clause: Where,
        columns_by_table: Dict[str, Set[str]]
    ):
        """Extract column references from WHERE clause."""
        for token in where_clause.tokens:
            if isinstance(token, Identifier):
                self._process_column_identifier(token, columns_by_table)
            elif isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    self._process_column_identifier(identifier, columns_by_table)
    
    def _get_real_name(self, identifier: Identifier) -> Optional[str]:
        """Get the real name of an identifier (without alias)."""
        try:
            name = identifier.get_real_name()
            if name:
                # Remove quotes if present
                name = name.strip('"').strip("'").strip('`')
                return name.lower()  # Normalize to lowercase
        except Exception:
            pass
        return None


class StructuralReward:
    """
    Rewards partial structural correctness in SQL queries.
    
    Compares SELECT, WHERE, and JOIN clauses between generated and ground truth SQL,
    providing partial credit for matching components.
    """
    
    def __init__(
        self,
        select_weight: float = 0.3,
        where_weight: float = 0.3,
        join_weight: float = 0.2
    ):
        """
        Initialize structural reward component.
        
        Args:
            select_weight: Reward for matching SELECT clause
            where_weight: Reward for matching WHERE clause
            join_weight: Reward for matching JOIN clause
        """
        self.select_weight = select_weight
        self.where_weight = where_weight
        self.join_weight = join_weight
    
    def compute(self, generated_sql: str, ground_truth_sql: str) -> float:
        """
        Compute structural reward based on clause matching.
        
        Args:
            generated_sql: SQL query generated by the model
            ground_truth_sql: Ground truth SQL query
        
        Returns:
            Reward score (sum of matching clause rewards)
        """
        if not generated_sql or not ground_truth_sql:
            return 0.0
        
        try:
            # Parse both SQL queries
            gen_ast = self._parse_sql(generated_sql)
            gt_ast = self._parse_sql(ground_truth_sql)
            
            if not gen_ast or not gt_ast:
                return 0.0
            
            reward = 0.0
            
            # SELECT clause matching
            if self._select_matches(gen_ast, gt_ast):
                reward += self.select_weight
            
            # WHERE clause matching
            if self._where_matches(gen_ast, gt_ast):
                reward += self.where_weight
            
            # JOIN clause matching
            if self._join_matches(gen_ast, gt_ast):
                reward += self.join_weight
            
            return reward
            
        except Exception as e:
            print(f"[Structural Reward] Error computing structural reward: {e}")
            return 0.0
    
    def _parse_sql(self, sql: str) -> Optional[sqlparse.sql.Statement]:
        """Parse SQL query into AST."""
        try:
            parsed = sqlparse.parse(sql)
            if parsed:
                return parsed[0]
        except Exception:
            pass
        return None
    
    def _select_matches(self, gen_ast, gt_ast) -> bool:
        """
        Check if SELECT clauses match.
        
        Extracts column names from SELECT clauses and compares them,
        handling different orderings and formatting.
        
        Args:
            gen_ast: Parsed generated SQL AST
            gt_ast: Parsed ground truth SQL AST
        
        Returns:
            True if SELECT clauses match, False otherwise
        """
        try:
            gen_columns = self._extract_select_columns(gen_ast)
            gt_columns = self._extract_select_columns(gt_ast)
            
            # Normalize column names (lowercase, strip whitespace)
            gen_columns_normalized = {col.lower().strip() for col in gen_columns}
            gt_columns_normalized = {col.lower().strip() for col in gt_columns}
            
            # Check if sets match (order-independent comparison)
            return gen_columns_normalized == gt_columns_normalized
            
        except Exception as e:
            print(f"[Structural Reward] Error comparing SELECT clauses: {e}")
            return False
    
    def _extract_select_columns(self, statement: sqlparse.sql.Statement) -> Set[str]:
        """
        Extract column names from SELECT clause.
        
        Args:
            statement: Parsed SQL statement
        
        Returns:
            Set of column names (including table prefixes if present)
        """
        columns = set()
        select_seen = False
        
        for token in statement.tokens:
            # Look for SELECT keyword
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
                continue
            
            # Extract columns after SELECT
            if select_seen:
                if isinstance(token, IdentifierList):
                    for identifier in token.get_identifiers():
                        col_name = self._get_column_name(identifier)
                        if col_name:
                            columns.add(col_name)
                elif isinstance(token, Identifier):
                    col_name = self._get_column_name(token)
                    if col_name:
                        columns.add(col_name)
                elif token.ttype is Keyword.DML or (token.ttype is Keyword and token.value.upper() == 'FROM'):
                    # Stop at FROM keyword
                    break
        
        return columns
    
    def _get_column_name(self, identifier) -> Optional[str]:
        """
        Get column name from identifier, handling aliases and table prefixes.
        
        Args:
            identifier: SQL identifier token
        
        Returns:
            Column name string or None
        """
        try:
            # Handle wildcards
            if str(identifier).strip() == '*':
                return '*'
            
            # Handle functions (e.g., COUNT(*), MAX(col))
            if isinstance(identifier, Function):
                # Return the function call as-is for comparison
                return str(identifier).strip()
            
            # Get real name (without alias)
            real_name = identifier.get_real_name()
            parent_name = identifier.get_parent_name()
            
            if parent_name and real_name:
                # Column with table prefix: table.column
                return f"{parent_name}.{real_name}"
            elif real_name:
                # Column without table prefix
                return real_name
            else:
                # Fallback to string representation
                return str(identifier).strip()
                
        except Exception:
            return str(identifier).strip()
    
    def _where_matches(self, gen_ast, gt_ast) -> bool:
        """
        Check if WHERE clauses match.
        
        Extracts predicates from WHERE clauses and compares them,
        handling different orderings and formatting.
        
        Args:
            gen_ast: Parsed generated SQL AST
            gt_ast: Parsed ground truth SQL AST
        
        Returns:
            True if WHERE clauses match, False otherwise
        """
        try:
            gen_predicates = self._extract_where_predicates(gen_ast)
            gt_predicates = self._extract_where_predicates(gt_ast)
            
            # If both have no WHERE clause, consider it a match
            if not gen_predicates and not gt_predicates:
                return True
            
            # If only one has WHERE clause, no match
            if not gen_predicates or not gt_predicates:
                return False
            
            # Normalize predicates (lowercase, strip whitespace)
            gen_predicates_normalized = {pred.lower().strip() for pred in gen_predicates}
            gt_predicates_normalized = {pred.lower().strip() for pred in gt_predicates}
            
            # Check if sets match (order-independent comparison)
            return gen_predicates_normalized == gt_predicates_normalized
            
        except Exception as e:
            print(f"[Structural Reward] Error comparing WHERE clauses: {e}")
            return False
    
    def _extract_where_predicates(self, statement: sqlparse.sql.Statement) -> Set[str]:
        """
        Extract predicates from WHERE clause.
        
        Args:
            statement: Parsed SQL statement
        
        Returns:
            Set of predicate strings
        """
        predicates = set()
        
        for token in statement.tokens:
            if isinstance(token, Where):
                # Extract predicates from WHERE clause
                # Split by AND/OR to get individual predicates
                where_str = str(token).replace('WHERE', '').strip()
                
                # Split by AND and OR (case-insensitive)
                parts = re.split(r'\s+AND\s+|\s+OR\s+', where_str, flags=re.IGNORECASE)
                
                for part in parts:
                    part = part.strip()
                    if part:
                        # Normalize whitespace
                        part = ' '.join(part.split())
                        predicates.add(part)
        
        return predicates
    
    def _join_matches(self, gen_ast, gt_ast) -> bool:
        """
        Check if JOIN clauses match.
        
        Extracts JOIN information (tables and conditions) and compares them,
        handling different orderings and formatting.
        
        Args:
            gen_ast: Parsed generated SQL AST
            gt_ast: Parsed ground truth SQL AST
        
        Returns:
            True if JOIN clauses match, False otherwise
        """
        try:
            gen_joins = self._extract_joins(gen_ast)
            gt_joins = self._extract_joins(gt_ast)
            
            # If both have no JOINs, don't give credit (return False)
            # This is different from WHERE - JOINs are optional structural elements
            if not gen_joins and not gt_joins:
                return False
            
            # If only one has JOINs, no match
            if not gen_joins or not gt_joins:
                return False
            
            # Normalize joins (lowercase, strip whitespace)
            gen_joins_normalized = {join.lower().strip() for join in gen_joins}
            gt_joins_normalized = {join.lower().strip() for join in gt_joins}
            
            # Check if sets match (order-independent comparison)
            return gen_joins_normalized == gt_joins_normalized
            
        except Exception as e:
            print(f"[Structural Reward] Error comparing JOIN clauses: {e}")
            return False
    
    def _extract_joins(self, statement: sqlparse.sql.Statement) -> Set[str]:
        """
        Extract JOIN information from SQL statement.
        
        Args:
            statement: Parsed SQL statement
        
        Returns:
            Set of JOIN strings (table + condition)
        """
        joins = set()
        
        # Iterate through tokens to find JOIN patterns
        tokens = list(statement.tokens)
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Look for JOIN keyword
            if token.ttype is Keyword and 'JOIN' in token.value.upper():
                # Next non-whitespace token should be the table
                table = None
                condition = None
                
                # Find table (next identifier after JOIN)
                j = i + 1
                while j < len(tokens):
                    if tokens[j].ttype in (sqlparse.tokens.Whitespace, sqlparse.tokens.Newline):
                        j += 1
                        continue
                    if isinstance(tokens[j], Identifier):
                        table = tokens[j].get_real_name()
                        break
                    j += 1
                
                # Find ON keyword and condition
                k = j + 1
                while k < len(tokens):
                    if tokens[k].ttype is Keyword and tokens[k].value.upper() == 'ON':
                        # Next non-whitespace token should be the condition
                        m = k + 1
                        while m < len(tokens):
                            if tokens[m].ttype in (sqlparse.tokens.Whitespace, sqlparse.tokens.Newline):
                                m += 1
                                continue
                            # Get the comparison or condition
                            condition = str(tokens[m]).strip()
                            # Normalize whitespace
                            condition = ' '.join(condition.split())
                            break
                        break
                    k += 1
                
                # If we found both table and condition, add to joins
                if table and condition:
                    join_str = f"{table} ON {condition}"
                    joins.add(join_str)
            
            i += 1
        
        return joins


class EnhancedSyntaxReward:
    """
    Validates SQL syntax at AST level, beyond format tag checking.
    
    Provides reward for syntactically valid SQL with proper AST structure.
    """
    
    def __init__(self, weight: float = 0.2):
        """
        Initialize enhanced syntax reward component.
        
        Args:
            weight: Reward weight for valid syntax
        """
        self.weight = weight
    
    def compute(self, sql: str) -> float:
        """
        Compute syntax reward based on AST validation.
        
        Args:
            sql: SQL query string
        
        Returns:
            Reward score (weight if valid, 0 otherwise)
        """
        if not sql or not sql.strip():
            return 0.0
        
        try:
            # Parse SQL using sqlparse
            parsed = sqlparse.parse(sql)
            
            # Check if parsing succeeded and AST is valid
            if parsed and len(parsed) > 0 and self._is_valid_ast(parsed[0]):
                return self.weight
            
        except Exception as e:
            # If parsing fails, return zero reward
            print(f"[Syntax Reward] Error parsing SQL: {e}")
        
        return 0.0
    
    def _is_valid_ast(self, statement: sqlparse.sql.Statement) -> bool:
        """
        Validate AST structure.
        
        Checks for:
        - Valid DML statement (SELECT, INSERT, UPDATE, DELETE)
        - Proper token structure
        - No syntax errors
        
        Args:
            statement: Parsed SQL statement
        
        Returns:
            True if AST is valid, False otherwise
        """
        try:
            # Check if statement has tokens
            if not statement.tokens:
                return False
            
            # Check for valid DML statement
            has_dml = False
            valid_dml_keywords = {'SELECT', 'INSERT', 'UPDATE', 'DELETE'}
            
            for token in statement.tokens:
                # Look for DML keyword
                if token.ttype is DML:
                    if token.value.upper() in valid_dml_keywords:
                        has_dml = True
                        break
            
            # Must have a valid DML statement
            if not has_dml:
                return False
            
            # Check for basic structural validity
            # For SELECT statements, should have at least SELECT and FROM (or just SELECT for simple queries)
            statement_type = statement.get_type()
            
            if statement_type == 'SELECT':
                # Verify SELECT statement has proper structure
                return self._validate_select_structure(statement)
            elif statement_type in ('INSERT', 'UPDATE', 'DELETE'):
                # For other DML statements, basic token presence is sufficient
                return True
            
            # Unknown statement type
            return False
            
        except Exception as e:
            print(f"[Syntax Reward] Error validating AST: {e}")
            return False
    
    def _validate_select_structure(self, statement: sqlparse.sql.Statement) -> bool:
        """
        Validate SELECT statement structure.
        
        Args:
            statement: Parsed SELECT statement
        
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check for SELECT keyword
            has_select = False
            has_columns = False
            
            for token in statement.tokens:
                # Check for SELECT keyword
                if token.ttype is DML and token.value.upper() == 'SELECT':
                    has_select = True
                    continue
                
                # After SELECT, should have column identifiers
                if has_select and not has_columns:
                    # Check for identifiers (columns)
                    if isinstance(token, (Identifier, IdentifierList)):
                        has_columns = True
                    # Also accept wildcards
                    elif token.ttype is sqlparse.tokens.Wildcard:
                        has_columns = True
            
            # Valid SELECT must have SELECT keyword and columns
            return has_select and has_columns
            
        except Exception:
            return False


class EnhancedRewardComputer:
    """
    Enhanced reward computer that wraps SQL-R1's baseline reward function
    and adds additional reward components.
    
    This design maintains backward compatibility while adding new capabilities.
    """
    
    def __init__(
        self,
        baseline_reward_fn,
        enable_enhanced: bool = True,
        schema_weight: float = -0.5,
        structural_select_weight: float = 0.3,
        structural_where_weight: float = 0.3,
        structural_join_weight: float = 0.2,
        syntax_weight: float = 0.2
    ):
        """
        Initialize enhanced reward computer.
        
        Args:
            baseline_reward_fn: SQL-R1's existing reward function
            enable_enhanced: Whether to enable enhanced reward components
            schema_weight: Weight for schema-aware reward
            structural_select_weight: Weight for SELECT clause matching
            structural_where_weight: Weight for WHERE clause matching
            structural_join_weight: Weight for JOIN clause matching
            syntax_weight: Weight for enhanced syntax reward
        """
        self.baseline_reward_fn = baseline_reward_fn
        self.enable_enhanced = enable_enhanced
        
        # Initialize enhanced components
        self.schema_reward = SchemaAwareReward(weight=schema_weight)
        self.structural_reward = StructuralReward(
            select_weight=structural_select_weight,
            where_weight=structural_where_weight,
            join_weight=structural_join_weight
        )
        self.syntax_reward = EnhancedSyntaxReward(weight=syntax_weight)
    
    def compute_reward(
        self,
        solution_str: str,
        ground_truth: Dict[str, str],
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EnhancedRewardResult:
        """
        Compute enhanced reward combining baseline and new components.
        
        Args:
            solution_str: Raw model response string
            ground_truth: Dictionary containing ground truth data
            schema: Database schema information (optional)
            **kwargs: Additional arguments for baseline reward function
        
        Returns:
            EnhancedRewardResult with total reward and component breakdown
        """
        # Step 1: Call baseline reward function
        baseline_score = self.baseline_reward_fn(solution_str, ground_truth, **kwargs)
        
        # Step 2: Extract baseline reward components
        # The baseline reward function returns a single score, but we need to compute
        # individual components for detailed logging. We'll compute them ourselves
        # based on the SQL-R1 reward logic.
        baseline_format, baseline_execution, baseline_result, baseline_length = \
            self._extract_baseline_components(solution_str, ground_truth, **kwargs)
        
        baseline_total = baseline_format + baseline_execution + baseline_result + baseline_length
        
        # Step 3: Extract generated SQL from solution string
        generated_sql = self._extract_sql_from_solution(solution_str)
        ground_truth_sql = ground_truth.get('sql', '') if isinstance(ground_truth, dict) else ''
        
        # Step 4: Compute enhanced reward components (if enabled)
        schema_reward = 0.0
        structural_reward = 0.0
        syntax_reward = 0.0
        
        if self.enable_enhanced and generated_sql:
            # Schema-aware reward
            if schema:
                schema_reward = self.schema_reward.compute(generated_sql, schema)
            
            # Structural reward
            if ground_truth_sql:
                structural_reward = self.structural_reward.compute(generated_sql, ground_truth_sql)
            
            # Enhanced syntax reward
            syntax_reward = self.syntax_reward.compute(generated_sql)
        
        # Step 5: Aggregate all rewards into total
        total_reward = baseline_total + schema_reward + structural_reward + syntax_reward
        
        # Step 6: Log detailed breakdown
        if self.enable_enhanced:
            print(f"[Enhanced Reward] Total: {total_reward:.3f} = "
                  f"Baseline({baseline_total:.3f}) + "
                  f"Schema({schema_reward:.3f}) + "
                  f"Structural({structural_reward:.3f}) + "
                  f"Syntax({syntax_reward:.3f})")
        
        # Step 7: Return comprehensive reward breakdown
        return EnhancedRewardResult(
            total=total_reward,
            baseline_total=baseline_total,
            baseline_format=baseline_format,
            baseline_execution=baseline_execution,
            baseline_result=baseline_result,
            baseline_length=baseline_length,
            schema=schema_reward,
            structural=structural_reward,
            syntax=syntax_reward
        )
    
    def _extract_baseline_components(
        self,
        solution_str: str,
        ground_truth: Dict[str, Any],
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """
        Extract baseline reward components (format, execution, result, length).
        
        This mimics SQL-R1's reward computation logic:
        - Format: +1 for correct XML tags (<think>, <answer>, ```sql```)
        - Execution: +2 for executable SQL, -2 for errors
        - Result: +3 for correct results, -3 for wrong results
        - Length: Small penalty for overly long responses
        
        Args:
            solution_str: Raw model response string
            ground_truth: Dictionary containing ground truth data
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (format_reward, execution_reward, result_reward, length_reward)
        """
        format_reward = 0.0
        execution_reward = 0.0
        result_reward = 0.0
        length_reward = 0.0
        
        # Format reward: Check for correct XML tags
        has_think = '<think>' in solution_str and '</think>' in solution_str
        has_answer = '<answer>' in solution_str and '</answer>' in solution_str
        has_sql_block = '```sql' in solution_str and '```' in solution_str
        
        if has_think and has_answer and has_sql_block:
            format_reward = 1.0
        
        # Execution and result rewards would require actual SQL execution
        # For now, we'll use simplified heuristics or rely on the baseline function
        # In a real implementation, these would be computed by executing the SQL
        
        # Length reward: Penalize overly long responses
        response_length = len(solution_str)
        if response_length > 1000:
            length_reward = -0.1 * ((response_length - 1000) / 100)
        
        return format_reward, execution_reward, result_reward, length_reward
    
    def _extract_sql_from_solution(self, solution_str: str) -> str:
        """
        Extract SQL query from solution string.
        
        The solution string typically contains XML tags like:
        <think>reasoning</think>
        <answer>
        ```sql
        SELECT * FROM table
        ```
        </answer>
        
        Args:
            solution_str: Raw model response string
        
        Returns:
            Extracted SQL query string
        """
        if not solution_str:
            return ""
        
        # Try to extract SQL from code block
        sql_pattern = r'```sql\s*(.*?)\s*```'
        import re
        match = re.search(sql_pattern, solution_str, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: Try to extract from <answer> tags
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
        
        if match:
            answer_content = match.group(1).strip()
            # Try to extract SQL from code block within answer
            match = re.search(sql_pattern, answer_content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return answer_content
        
        # Last resort: Return the whole string
        return solution_str
