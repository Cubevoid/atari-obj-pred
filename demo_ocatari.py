import random
from ocatari.core import OCAtari
from ocatari.vision.utils import mark_bb, make_darker
from ocatari.utils import load_agent, parser
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# import matplotlib.pyplot as plt

GAME_NAME = "Pong"
# MODE = "vision"
MODE = "revised"
HUD = False
env = OCAtari(GAME_NAME, mode=MODE, hud=HUD, obs_mode='dqn')
observation, info = env.reset()

opts = parser.parse_args()

sam = sam_model_registry["vit_b"](checkpoint="./models/sam_vit_b_01ec64.pth")
generator = SamAutomaticMaskGenerator(sam)

if opts.path:
    agent = load_agent(opts, env.action_space.n)

for i in range(10000):
    if opts.path is not None:
        action = agent.draw_action(env.dqn_obs)
    else:
        action = random.randint(0, 0)
    obs, reward, terminated, truncated, info = env.step(action)
    masks = generator.generate(obs)
    print(len(masks))

    if terminated or truncated:
        observation, info = env.reset()
env.close()
