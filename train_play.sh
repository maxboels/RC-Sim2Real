
#
python isaac_lab/scripts/reinforcement_learning/skrl/train.py --task Isaac-QuantumTracer-Direct-v0 --num_envs 32

# headless
python isaac_lab/scripts/reinforcement_learning/skrl/train.py --task Isaac-QuantumTracer-Direct-v0 --num_envs 4096 --headless

# playing
python isaac_lab/scripts/reinforcement_learning/skrl/play.py --task Isaac-QuantumTracer-Direct-v0 --num_envs 32


# bash version
cd isaac_lab
./isaaclab.sh -p source/standalone/workflows/skrl/train.py \
    --task Isaac-Quantum-Tracer-Direct-v0 \
    --num_envs 4096