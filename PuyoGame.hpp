#pragma once
#include "define.hpp"

namespace puyoGame
{
  static const VVI ACTIONSDICT = {
      {1, 1, 0, 0, 0, 0},
      {2, 0, 0, 0, 0, 0},
      {2, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0},
      {0, 2, 0, 0, 0, 0},
      {0, 1, 1, 0, 0, 0},
      {0, 2, 0, 0, 0, 0},
      {0, 1, 1, 0, 0, 0},
      {0, 0, 2, 0, 0, 0},
      {0, 0, 1, 1, 0, 0},
      {0, 0, 2, 0, 0, 0},
      {0, 0, 1, 1, 0, 0},
      {0, 0, 0, 2, 0, 0},
      {0, 0, 0, 1, 1, 0},
      {0, 0, 0, 2, 0, 0},
      {0, 0, 0, 1, 1, 0},
      {0, 0, 0, 0, 2, 0},
      {0, 0, 0, 0, 1, 1},
      {0, 0, 0, 0, 2, 0},
      {0, 0, 0, 0, 1, 1},
      {0, 0, 0, 0, 0, 2},
      {0, 0, 0, 0, 0, 2}};

  class Puyo
  {
  public:
    static const int NONE = 0;
    static const int RED = 1;
    static const int BLUE = 2;
    static const int GREEN = 3;
    static const int YELLOW = 4;
    static const int OJAMA = 5;

    static const int LEFT = 1;
    static const int UP = 2;
    static const int RIGHT = 3;
    static const int DOWN = 4;

    int color1, color2, direction;

    Puyo(int color1, int color2);
    void set_direct(int direct);
  };

  class State
  {
  public:
    VVI gameMap;
    VVI puyos;
    int turn;

    State();
    State(VVI gameMap, VVI puyos, int turn);
    bool isDone();
    bool isLose();
    bool isEnd();

    Puyo getNextPuyo();

    VVI oneFall(VVI stage, size_t x, int color, bool &isAlive);
    VVI puyoFall(VVI stage, size_t x, Puyo puyo, bool &isAlive);
    VVI erase(VVI stage, VVI newStage, int &getScore);
    VVI erasePuyo(VVI stage, size_t x, size_t y, int color, int &counter);
    VVI eraseSimulation(VVI stage, VVI newStage, int &reward);
    VVI fall(VVI stage);

    VI legalActions();
    State next(int action, VI nextPuyoColors, int &reward);
    int calcMaxReward();
  };
} // namespace puyoGame
