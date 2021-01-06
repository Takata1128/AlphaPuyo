#pragma once
#include <vector>

//ゲーム定義
#define GAMEMAP_HEIGHT 7
#define GAMEMAP_WIDTH 4
#define ACTION_KIND (2 * 3 + (GAMEMAP_WIDTH - 2) * 4)
#define PUYO_COLOR 3
#define MAX_STEP 25
#define DEAD_X 1
#define DEAD_Y 1

// for MCTS
#define SINGLE 0
#define RANDOM 1

#define EVALUATE_DEPTH 10
#define TSUMO_SIZE 11
#define TRY_COUNT 20
#define PV_EVALUATE_COUNT 1000
#define C_PUCT 15.0
#define MCTS_TYPE SINGLE
#define MCTS_TEMPERATURE 1.0

// for construction of network
#define FIELD_CHANNELS (PUYO_COLOR + 1)
#define PUYO_CHANNELS (PUYO_COLOR * 4)
#define TURN_CHANNELS 6
#define CHANNEL_SIZE (FIELD_CHANNELS + PUYO_CHANNELS)
#define DN_OUTPUT_SIZE ACTION_KIND

// for selfplay
#define SP_GAME_COUNT 100

// for evaluate network
#define EN_GAME_COUNT 50

// for multithread
#define THREAD_NUM 5

using VI = std::vector<int>;
using VVI = std::vector<std::vector<int>>;
