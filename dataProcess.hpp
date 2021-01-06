#pragma once
#include "PuyoGame.hpp"
#include "define.hpp"
#include <vector>

namespace dataprocess {
std::vector<float> data2Binary(const VVI &stage, VI nowPuyo, VI nextPuyo,
                               int turn) {
    std::vector<std::vector<std::vector<int>>> mat(
        GAMEMAP_HEIGHT, std::vector<std::vector<int>>(
                            GAMEMAP_WIDTH, std::vector<int>(CHANNEL_SIZE)));
    for(int i = 0; i < GAMEMAP_HEIGHT; i++)
        for(int j = 0; j < GAMEMAP_WIDTH; j++)
            mat[i][j][stage[i][j]] = 1;

    VI puyoData = VI(4);
    for(int i = 0; i < 2; i++)
        puyoData[i] = nowPuyo[i];
    for(int i = 0; i < 2; i++)
        puyoData[i + 2] = nextPuyo[i];
    for(int k = 0; k < 4; k++)
        for(int i = 0; i < GAMEMAP_HEIGHT; i++)
            for(int j = 0; j < GAMEMAP_WIDTH; j++)
                mat[i][j][puyoData[k] + PUYO_COLOR + k * PUYO_COLOR] = 1;
    std::vector<float> ret(GAMEMAP_HEIGHT * GAMEMAP_WIDTH * CHANNEL_SIZE);
    int cur = 0;
    for(int i = 0; i < GAMEMAP_HEIGHT; i++)
        for(int j = 0; j < GAMEMAP_WIDTH; j++)
            for(int k = 0; k < CHANNEL_SIZE; k++)
                ret[cur++] = mat[i][j][k];
    return ret;
}

std::vector<float> encode(const puyogame::State &state) {
    return data2Binary(state.gameMap, state.puyos[0], state.puyos[1],
                       state.turn);
}

} // namespace dataprocess