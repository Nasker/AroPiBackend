import sqlite3
from typing import List, Tuple
from config import DB_PATH


class CombinationGenerator:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
    
    def _get_pictos_by_grammar_type(self, grammar_type: str, language: str = None) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if language:
            cursor.execute(
                "SELECT word FROM pictos WHERE grammar_type = ? AND language = ?",
                (grammar_type, language)
            )
        else:
            cursor.execute(
                "SELECT word FROM pictos WHERE grammar_type = ?",
                (grammar_type,)
            )
        
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    def generate_pronoun_verb_combinations(self, language: str = None) -> List[List[str]]:
        pronouns = self._get_pictos_by_grammar_type("pronoun", language)
        verbs = self._get_pictos_by_grammar_type("verb", language)
        
        combinations = []
        for pronoun in pronouns:
            for verb in verbs:
                combinations.append([pronoun, verb])
        
        return combinations
    
    def get_all_grammar_types(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT grammar_type FROM pictos WHERE grammar_type IS NOT NULL")
        types = [row[0] for row in cursor.fetchall()]
        conn.close()
        return types
    
    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM pictos")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT grammar_type, COUNT(*) FROM pictos GROUP BY grammar_type")
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT language, COUNT(*) FROM pictos GROUP BY language")
        by_language = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_pictos": total,
            "by_grammar_type": by_type,
            "by_language": by_language
        }


if __name__ == "__main__":
    generator = CombinationGenerator()
    
    print("Database Statistics:")
    stats = generator.get_stats()
    print(f"Total pictos: {stats['total_pictos']}")
    print(f"By grammar type: {stats['by_grammar_type']}")
    print(f"By language: {stats['by_language']}")
    print()
    
    print("Available grammar types:")
    print(generator.get_all_grammar_types())
    print()
    
    combinations = generator.generate_pronoun_verb_combinations()
    print(f"Generated {len(combinations)} pronoun + verb combinations")
    print("\nFirst 100 combinations:")
    for combo in combinations[:100]:
        print(f"  {combo}")
