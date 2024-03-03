import random
from ocatari.core import OCAtari
from ocatari.vision.utils import mark_bb, make_darker
from ocatari.utils import load_agent, parser
# import matplotlib.pyplot as plt

GAME_NAME = "Assault"
# MODE = "vision"
MODE = "revised"
HUD = False
env = OCAtari(GAME_NAME, mode=MODE, hud=HUD, render_mode='human')
observation, info = env.reset()

opts = parser.parse_args()

if opts.path:
    agent = load_agent(opts, env.action_space.n)

for i in range(10000):
    if opts.path is not None:
        action = agent.draw_action(env.dqn_obs)
    else:
        action = random.randint(0, 0)
    obs, reward, terminated, truncated, info = env.step(action)

    if i % 10 == 0:
        print(env.objects)
        for obj in env.objects:
            x, y = obj.xy
            if x < 160 and y < 210:
                opos = obj.xywh
                ocol = obj.rgb
                sur_col = make_darker(ocol)
                mark_bb(obs, opos, color=sur_col)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()
env.close()
