import sys
from pathlib import Path

def make_briefing():
    print("="*60)
    print(" BMVC 2026: MASTER EVALUATION BRIEFING ")
    print("="*60)
    
    print("\n[STEP 1: ACCURACY & STABILITY]")
    print("Goal: Fill Table 3-b (Primary Benchmark)")
    print("Command (Hamming): python tools/aggregate_final_metrics.py")
    print("Required data: 10-seed average mAP@.5 ± std.")
    
    print("\n[STEP 2: NIGHT/DAY & CLASS-WISE]")
    print("Goal: Fill Table 6, Table 7, and Generalization Section")
    print("Command (Hamming): sbatch slurm/run_paper_tables.sh")
    print("Required data: Class-wise APs, Recall, Miss Rate for Day vs Night.")
    
    print("\n[STEP 3: EFFICIENCY]")
    print("Goal: Fill Table 3-b Efficiency / Section 4.2")
    print("Command (Hamming): python tools/get_efficiency.py")
    print("Required data: GFLOPs, Parameter count, Inference Latency.")
    
    print("\n[STEP 4: QUALITY CHECK]")
    print("Goal: Verify FLIR Day/Night split integrity")
    print("Command (Hamming): python tools/analyze_night.py")
    print("Note: If 'Night' is 0, consider lowering threshold in tools/split_day_night.py")
    
    print("\n" + "="*60)
    print(" NEXT STEPS FOR SUBMISSION ")
    print("="*60)
    print("1. Run Step 2 (sbatch) to get the bulk of the numbers.")
    print("2. Run Step 3 to confirm GFLOPs parity with the 190.3 measurement.")
    print("3. Check 'logs/paper_tables.out' for the copy-pasteable tables.")
    print("4. Replace draft numbers in 'bmvc_review_v0_1.tex' using search/replace.")
    
if __name__ == "__main__":
    make_briefing()
