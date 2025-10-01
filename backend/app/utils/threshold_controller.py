# File: .\backend\app\utils\threshold_controller.py 

import numpy as np
import json
from typing import Dict, Any, List
from datetime import datetime
from app.utils.db import DatabaseManager


class ThresholdController:
    def __init__(self, window_k: int = 20, p_up: float = 0.8, p_relax: float = 0.5, delta_tighten: float = 0.05, delta_relax: float = 0.05):
        self.db = DatabaseManager ()
        print("ThresholdController initialized and connected to DB.")
        self.window_k = window_k
        self.p_up = p_up
        self.p_relax = p_relax
        self.delta_tighten = delta_tighten
        self.delta_relax = delta_relax

    def get_current_thresholds(self, user_id: str, exercise_id: str) -> Dict[str, float]:
        """Retrieves active thresholds for a user and exercise."""
        # เรียกใช้ db.get_user_thresholds
        return self.db.get_user_thresholds(user_id, exercise_id)

    def log_rep_result(self, user_id: str, exercise_id: str, is_success: bool, metric_errors: Dict[str, float]):
        """Logs single rep result to history table."""
        record = {
            "user_id": user_id,
            "exercise_id": exercise_id,
            "is_success": is_success,
            "metric_errors": metric_errors # JSONB field
        }
        self.db.log_rep_result(record)
        
    def propose_new_thresholds(self, user_id: str, exercise_id: str) -> Dict[str, float]:
        """
        Calculates and proposes new personalized thresholds based on recent performance.
        Logs the proposal before returning.
        """
        history = self.db.fetch_rep_history(user_id, exercise_id, self.window_k)
        if len(history) < self.window_k:
            return {} 

        success_count = sum(1 for is_success, _ in history if is_success)
        success_rate = success_count / self.window_k

        current_thresholds = self.get_current_thresholds(user_id, exercise_id)
        proposed_updates = {}

        for metric_name, current_threshold in current_thresholds.items():
            if success_rate >= self.p_up:
                # Tighten (ปรับให้ยากขึ้น)
                proposed_updates[metric_name] = current_threshold * (1.0 - self.delta_tighten)
            elif success_rate <= self.p_relax:
                # Relax (ปรับให้ง่ายขึ้น)
                proposed_updates[metric_name] = current_threshold * (1.0 + self.delta_relax)
        
        proposal_record = {
            "user_id": user_id,
            "exercise_id": exercise_id,
            "proposed_thresholds": proposed_updates # Pass the dictionary directly
        }
        self.db.log_threshold_proposal(proposal_record) # บันทึก

        return proposed_updates

    def commit_thresholds(self, user_id: str, exercise_id: str, updates: Dict[str, float]):
        """Commits the proposed thresholds after user confirmation."""
        self.db.update_user_thresholds(user_id, exercise_id, updates)