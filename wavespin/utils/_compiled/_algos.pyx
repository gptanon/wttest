import cython


float_int = cython.fused_type(cython.floating, cython.integral)


@cython.wraparound(False)
cpdef int smallest_interval_over_threshold(float_int[:] x, float_int threshold,
                                           int c=-1):
    # initialize variables ###################################################
    cdef Py_ssize_t N = x.shape[0]
    cdef float_int[:] x_view = x

    cdef Py_ssize_t right = 1
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t min_size = N
    cdef Py_ssize_t left_bound = N
    cdef float_int sm = x_view[left]

    # handle `left <= c < right` constraint
    if c != -1:
        # sum up to `c`
        for right in range(1, c + 1):
            sm += x_view[right]
        # constrain `left`, `right`
        left_bound = <Py_ssize_t>c
        right = <Py_ssize_t>(c + 1)

    # main loop ##############################################################
    # if `c == -1`, `left_bound` has no effect here
    while right < N and left <= left_bound:
        if sm > threshold:
            min_size = min(min_size, right - left)
            sm -= x_view[left]
            left += 1
        else:
            sm += x_view[right]
            right += 1

    # minimize from tail end. if `c == -1`, `left_bound` is `N`
    while left <= left_bound and sm > threshold:
        min_size = min(min_size, right - left)
        sm -= x[left]
        left += 1

    # return
    cdef int min_size_int = <int>min_size
    return min_size_int


@cython.wraparound(False)
cpdef tuple smallest_interval_over_threshold_indices(
    float_int[:] x, float_int threshold, int c, int interval):
    # initialize variables ###################################################
    cdef Py_ssize_t N = x.shape[0]
    cdef float_int[:] x_view = x

    # normally, this'd be `right = c + 1` (that's the lowest `right`, to include
    # `c`) and `left = right - interval`, but this allows `left < 0`, so do it
    # the other  way around
    cdef Py_ssize_t left = max(c + 1 - interval, 0)
    cdef Py_ssize_t right = left + interval
    # `left` cannot exceed `c`
    cdef Py_ssize_t left_end_tentative = min(c, left + interval)
    # `right` cannot exceed `N`
    cdef Py_ssize_t right_end_tentative = min(N, right + interval)
    # the two bounds are independent so account for both
    cdef Py_ssize_t max_sweep = min(left_end_tentative - left,
                                    right_end_tentative - right) + 1
    cdef float_int sm = x_view[left]

    # edge case: due to setup of the main loop, this will out of bound
    # per `right`, and accounting for this may require inserting an additional
    # conditional in the main loop, which is slower, so do this instead
    if max_sweep + right > N:
        max_sweep -= 1

    # initial sum
    cdef Py_ssize_t idx = 0
    for idx in range(left + 1, right):
        sm += x_view[idx]

    # main loop variables
    cdef double dist_to_c = -1.
    cdef double dist_to_c_min = <double>N
    cdef double midpt = 0.
    cdef double cdouble = <double>c
    cdef Py_ssize_t the_left = 0
    cdef Py_ssize_t the_right = 0
    cdef Py_ssize_t shift = 0

    # main loop ##############################################################
    for shift in range(max_sweep):
        if sm > threshold:
            # maximum "closeness" is when `start, end` is centered around `c`,
            # and minimum is when `start` or `end` is at `c`
            midpt = <double>(left + right - 1) / 2.
            dist_to_c = abs(cdouble - midpt)
            # multiple matches, tiebreak by whatever best centers `c`
            if dist_to_c < dist_to_c_min:
                the_left = left
                the_right = right
                dist_to_c_min = min(dist_to_c_min, dist_to_c)
        sm += x_view[right] - x_view[left]
        right += 1
        left += 1
    else:
        # need one last check for `left` and `right` at bound
        if dist_to_c == -1 and sm > threshold:
            the_left = left
            the_right = right
        elif sm <= threshold and dist_to_c_min == <double>N:
            the_left = -1
            the_right = -1

    # return
    return (the_left, the_right)
