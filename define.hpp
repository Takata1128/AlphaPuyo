//ゲーム定義
#define GAMEMAP_HEIGHT 13
#define GAMEMAP_WIDTH 6
#define ACTION_KIND 22
#define PUYO_COLOR 4
#define MAX_STEP 50

// for MCTS
#define EVALUATE_DEPTH 5
#define TSUMO_SIZE 7
#define TRY_COUNT 10
#define PV_EVALUATE_COUNT 100
#define C_PUCT 5.0

// for construction of network
#define FIELD_CHANNELS 5
#define PUYO_CHANNELS 16
#define TURN_CHANNELS 6
#define CHANNEL_SIZE (FIELD_CHANNELS + PUYO_CHANNELS + TURN_CHANNELS)
#define DN_FILTERS 128
#define DN_RESIDUAL_NUM 5
#define DN_OUTPUT_SIZE ACTION_KIND

// for training network
#define RN_EPOCHS 20
#define BATCH_SIZE 128

// for selfplay
#define SP_GAME_COUNT 150
#define SP_TEMPERATURE 1.0

// for evaluate network
#define EN_GAME_COUNT 50
#define EN_TEMPERATURE 1.0

#define VI std::vector<int>
#define VVI std::vector<std::vector<int>>

