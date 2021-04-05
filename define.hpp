#pragma once
#include <vector>

//ゲーム定義
#define GAMEMAP_HEIGHT 13
#define GAMEMAP_WIDTH 6
#define ACTION_KIND (2 * 3 + (GAMEMAP_WIDTH - 2) * 4)
#define PUYO_COLOR 4
#define MAX_STEP 40
#define DEAD_X 2
#define DEAD_Y 1

// for MCTS
#define SINGLE 0
#define RANDOM 1

#define EVALUATE_DEPTH 100
#define TSUMO_SIZE MAX_STEP
#define TRY_COUNT 5
#define PV_EVALUATE_COUNT 100
#define C_PUCT 15.0
#define MCTS_TYPE RANDOM
#define MCTS_TEMPERATURE 1.0

// for construction of network
#define FIELD_CHANNELS (PUYO_COLOR + 1)
#define PUYO_CHANNELS (PUYO_COLOR * 4)
#define TURN_CHANNELS 6
#define CHANNEL_SIZE (FIELD_CHANNELS + PUYO_CHANNELS)
#define DN_OUTPUT_SIZE ACTION_KIND

// for selfplay
#define SP_GAME_COUNT 150

// for evaluate network
#define EN_GAME_COUNT 50

// for multithread
#define THREAD_NUM 5

using VI = std::vector<int>;
using VVI = std::vector<std::vector<int>>;
