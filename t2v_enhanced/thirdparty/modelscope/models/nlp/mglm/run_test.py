# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) 2022 Zhipu.AI

import sys

if sys.argv[1] == 'block':
    from test.test_block import main
    main()
elif sys.argv[1] == 'rel_shift':
    from test.test_rel_shift import main
    main()
