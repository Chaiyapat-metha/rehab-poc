# File: .\backend\app\utils\feedback_engine.py 

import random
from typing import Dict, Any, List

JOINT_NAMES_THAI = {
    0: "จมูก",
    1: "หัวตาซ้าย", 
    2: "ตาซ้าย", 
    3: "หางตาซ้าย",
    4: "หัวตาขวา",
    5: "ตาขวา",
    6: "หางตาขวา",
    7: "หูซ้าย",
    8: "หูขวา",
    9: "มุมปากซ้าย",
    10: "มุมปากขวา",
    11: "ไหล่ซ้าย",
    12: "ไหล่ขวา",
    13: "ข้อศอกซ้าย",
    14: "ข้อศอกขวา",
    15: "ข้อมือซ้าย",
    16: "ข้อมือขวา",
    17: "ปลายนิ้วโป้งซ้าย",
    18: "ปลายนิ้วโป้งขวา",
    19: "ปลายนิ้วก้อยซ้าย",
    20: "ปลายนิ้วก้อยขวา",
    21: "ปลายนิ้วชี้ซ้าย",
    22: "ปลายนิ้วชี้ขวา",
    23: "สะโพกซ้าย",
    24: "สะโพกขวา",
    25: "เข่าซ้าย",
    26: "เข่าขวา",
    27: "ข้อเท้าซ้าย",
    28: "ข้อเท้าขวา",
    29: "ส้นเท้าซ้าย",
    30: "ส้นเท้าขวา",
    31: "ปลายเท้าซ้าย",
    32: "ปลายเท้าขวา",
}

# 💡 Template Configuration (Req. 2)
FEEDBACK_TEMPLATES = {
    # {joint} -> ข้อศอก, เข่า
    # {status} -> งอไม่พอ, เหยียดมากไป
    
    # Template 1: Basic correction
    "basic": [
        "{joint} {status} นะคะ",
        "พยายามให้ {joint} {status} อีกนิดค่ะ"
    ],
    # Template 2: Specific error/advice
    "advice": [
        "{joint} {status} — พยายาม {advice_text}",
    ],
    # Status messages for specific joints
    "statuses": {
        'low': {
            'LEFT_KNEE': ["งอไม่พอ", "ยังไม่ค่อยลึก"],
            'RIGHT_KNEE': ["งอไม่พอ", "ยังไม่ค่อยลึก"],
            'LEFT_ELBOW': ["เหยียดมากไป", "ตึงเกินไป"],
            'RIGHT_ELBOW': ["เหยียดมากไป", "ตึงเกินไป"],
        },
        'high': {
            'LEFT_KNEE': ["งอมากเกินไป", "ลึกเกินไป"],
            'RIGHT_KNEE': ["งอมากเกินไป", "ลึกเกินไป"],
            'LEFT_ELBOW': ["งอน้อยไป", "ยังไม่เหยียดตรง"],
            'RIGHT_ELBOW': ["งอน้อยไป", "ยังไม่เหยียดตรง"],
        },
    },
    # Advice mapping (simplified example)
    "advice_map": {
        'LEFT_KNEE_low': "งอเข่าลงอีกนิด",
        'RIGHT_KNEE_low': "งอเข่าลงอีกนิด",
        'LEFT_ELBOW_high': "เหยียดแขนให้มากขึ้น",
        'RIGHT_ELBOW_high': "เหยียดแขนให้มากขึ้น",
    }
}

def map_joint_name(joint_name: str) -> str:
    """Maps internal joint names to Thai display names."""
    if 'KNEE' in joint_name:
        return 'เข่า' + ('ซ้าย' if 'LEFT' in joint_name else 'ขวา')
    if 'ELBOW' in joint_name:
        return 'ข้อศอก' + ('ซ้าย' if 'LEFT' in joint_name else 'ขวา')
    # Add other joints as needed
    return joint_name

def determine_joint_status(angle_mean: float, joint_name: str, thresholds: Dict[str, float]) -> str:
    """Determines if an angle error is 'low', 'high', or 'good' based on thresholds."""
    
    # Assume thresholds store the ideal angle and an allowed deviation (e.g., threshold_knee_min=80, threshold_knee_max=100)
    # Since the regression model predicts the mean, we check against the ideal range defined by the threshold module.
    
    # Example structure (You must implement the actual threshold logic in ThresholdController)
    nominal_angle = thresholds.get(f'{joint_name}_nominal', 90.0)
    tolerance = thresholds.get(f'{joint_name}_tolerance', 10.0) # Default tolerance 10 degrees
    
    if angle_mean < nominal_angle - tolerance:
        return 'low'
    elif angle_mean > nominal_angle + tolerance:
        return 'high'
    else:
        return 'good'

def generate_feedback(is_correct: bool, angle_errors_map: Dict[str, float], thresholds: Dict[str, float], exercise_id: str) -> Dict[str, Any]:
    """Generates display text, TTS text, and the list of joints to highlight."""
    
    if is_correct:
        return {
            "display_text": f"{exercise_id.replace('_', ' ')}: ท่าทางถูกต้อง!",
            "tts_text": "ยอดเยี่ยมค่ะ",
            "wrong_joints": []
        }
    
    wrong_joints = []
    error_details = []
    
    # 1. Identify critical errors based on angle regression
    for joint_name, angle_mean in angle_errors_map.items():
        status = determine_joint_status(angle_mean, joint_name, thresholds)
        
        if status != 'good':
            # Map joint name to index for highlighting
            if joint_name in JOINT_NAMES_THAI:
                wrong_joints.append(JOINT_NAMES_THAI[joint_name])
            
            # Select appropriate status message and template
            thai_joint = map_joint_name(joint_name)
            
            # Select random status phrase from pool
            status_phrase = random.choice(FEEDBACK_TEMPLATES["statuses"].get(status, {}).get(joint_name, ["ผิดพลาด"]))
            
            # Generate TTS text
            if f'{joint_name}_{status}' in FEEDBACK_TEMPLATES['advice_map']:
                # Use advice template
                advice_text = FEEDBACK_TEMPLATES['advice_map'][f'{joint_name}_{status}']
                tts_text_template = random.choice(FEEDBACK_TEMPLATES['advice'])
                
                tts_text = tts_text_template.format(
                    joint=thai_joint, 
                    status=status_phrase, 
                    advice_text=advice_text
                )
                display_text = f"{thai_joint} {status_phrase} ({advice_text})"
            else:
                # Use basic template
                tts_text_template = random.choice(FEEDBACK_TEMPLATES['basic'])
                tts_text = tts_text_template.format(
                    joint=thai_joint, 
                    status=status_phrase
                )
                display_text = f"{thai_joint} {status_phrase} (Angle: {angle_mean:.1f}°)"
                
            error_details.append({"display": display_text, "tts": tts_text, "priority": 1}) # Assign priority for selection

    # 2. Select the highest priority or combine if multiple (Req. 4)
    if not error_details:
        # Fallback if angle regression didn't detect critical error but classification failed
        return {
            "display_text": "ท่าทางผิดพลาดเล็กน้อย ลองปรับท่าทางดูค่ะ",
            "tts_text": "ลองปรับท่าทางดูนะคะ",
            "wrong_joints": [14] # Default fallback joint (Right Elbow)
        }
        
    # Simplify: Use the first detected error for TTS/Display
    top_error = error_details[0]
    
    return {
        "display_text": top_error['display'],
        "tts_text": top_error['tts'],
        # Ensure only unique joints are highlighted
        "wrong_joints": list(set(wrong_joints)) 
    }