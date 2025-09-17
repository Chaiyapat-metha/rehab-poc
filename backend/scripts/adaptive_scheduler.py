from datetime import datetime, timedelta
import sys
from pathlib import Path

# Setup path to import from app module
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.utils import db

# --- Simple Adaptive Logic Configuration ---
# If a user's max angle is > 105% of their current target, propose a new target.
PROPOSAL_THRESHOLD_MULTIPLIER = 1.05
# Propose the new target to be their recent max performance.
PROPOSAL_VALUE_SOURCE = "recent_max"

def get_current_targets(user_id: str) -> dict:
    """
    Fetches the current exercise targets for a user.
    In a real system, this would come from a dedicated 'user_targets' table.
    For this PoC, we will use a hardcoded default.
    """
    # TODO: Replace this with a database lookup
    return {
        'Incline rows with dumbbell': {
            'left_elbow_angle_max': 90.0,
            'right_elbow_angle_max': 90.0
        }
    }

def get_recent_performance(user_id: str) -> dict:
    """
    Fetches recent performance metrics for a user.
    For this PoC, we will simulate this by querying our 'frames' table.
    A real system would query an aggregated 'progress' table.
    """
    # TODO: In a real system, query the 'progress' table for weekly aggregated metrics.
    # This is a simplified query for demonstration.
    sql = """
    SELECT s.exercise_name, f.labels
    FROM frames f
    JOIN sessions s ON f.session_id = s.session_id
    WHERE s.user_id = %s AND f.labels IS NOT NULL;
    """
    performance = {}
    with db.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            results = cur.fetchall()
            for exercise_name, labels in results:
                # Simple aggregation: find the max severity value for low angle errors
                if labels.get('class') in [1, 2]: # Left/Right Elbow Low
                    severity = next(iter(labels.get('severity', {}).values()), 0.0)
                    if severity > 0:
                        # This logic is simplified. A real system would be more complex.
                        # For now, let's just track a max value.
                        if exercise_name not in performance:
                            performance[exercise_name] = {'left_elbow_angle_max': 0}
                        performance[exercise_name]['left_elbow_angle_max'] = max(
                            performance[exercise_name]['left_elbow_angle_max'], 90 + severity
                        )
    return performance

def main():
    print("--- Starting Adaptive Scheduler Batch Job ---")
    
    # 1. Get a list of all active users
    # For PoC, we'll just use our main trainer user
    all_users = ["ground_truth_trainer"] # In production, this would be `SELECT user_id FROM users;`

    for user_id in all_users:
        print(f"\nProcessing user: {user_id}")
        
        # 2. Get the user's current targets and recent performance
        current_targets = get_current_targets(user_id)
        recent_performance = get_recent_performance(user_id)

        # 3. Compare performance against targets and create proposals
        for exercise, targets in current_targets.items():
            if exercise not in recent_performance:
                continue

            for metric, target_value in targets.items():
                performance_value = recent_performance[exercise].get(metric)
                if not performance_value:
                    continue
                
                # 4. The Core Adaptive Logic
                if performance_value > target_value * PROPOSAL_THRESHOLD_MULTIPLIER:
                    print(f"  -> Proposal found for '{exercise}':")
                    print(f"     Metric '{metric}': Performance ({performance_value:.1f}) > Target ({target_value:.1f})")
                    
                    # 5. Insert the proposal into the database
                    proposal = {
                        "user_id": user_id,
                        "exercise_name": exercise,
                        "metric_key": metric,
                        "current_value": target_value,
                        "proposed_value": performance_value
                    }
                    db.insert_threshold_proposal(proposal)
                    print(f"     ✅ Proposal saved to database.")

    print("\n--- Adaptive Scheduler Job Finished ---")

if __name__ == "__main__":
    main()

