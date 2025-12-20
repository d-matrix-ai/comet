//gather, allgather, reduce


#include <iostream>
#include <cmath>
using namespace std;

/*GAHTER OPERATION*/
// Function to calculate total hops for row-wise gather
int calculate_row_hops(int n) {
    int total_hops = 0;
    for (int k = 0; (1 << k) < n; ++k) {
        total_hops += (1 << k);
    }
    return total_hops;
}

// Function to calculate total hops for column-wise gather
int calculate_col_hops(int m) {
    int total_hops = 0;
    for (int k = 0; (1 << k) < m; ++k) {
        total_hops += (1 << k);
    }
    return total_hops;
}

// Function to calculate total data moved for row-wise gather
int calculate_row_data_moved(int n, int data_size) {
    int total_data = 0;
    for (int k = 0; (1 << k) < n; ++k) {
        total_data += (1 << k) * data_size;
    }
    return total_data;
}

// Function to calculate total data moved for column-wise gather
int calculate_col_data_moved(int m, int data_size, int n) {
    int total_data = 0;
    for (int k = 0; (1 << k) < m; ++k) {
        total_data += (1 << k) * data_size * n;
    }
    return total_data;
}

int main() {
    int m, n, data_size;
    cout << "Enter number of rows (m): ";
    cin >> m;
    cout << "Enter number of columns (n): ";
    cin >> n;
    cout << "Enter data size per node: ";
    cin >> data_size;

    // Gather Phase
    int row_hops = m * calculate_row_hops(n);
    int col_hops = calculate_col_hops(m);

    int row_data_moved = m * calculate_row_data_moved(n, data_size);
    int col_data_moved = calculate_col_data_moved(m, data_size, n);

    int total_gather_hops = row_hops + col_hops;
    int total_gather_data_moved = row_data_moved + col_data_moved;

    cout << "\nTotal number of hops for Gather Operation: " << total_gather_hops << endl;
    cout << "Total data moved for Gather Operation: " << total_gather_data_moved << " units" << endl;

    return 0;
}


// Function to calculate total hops for row-wise gather
int calculate_row_hops(int n) {
    int total_hops = 0;
    for (int k = 0; (1 << k) < n; ++k) {
        total_hops += (1 << k);
    }
    return total_hops;
}

// Function to calculate total hops for column-wise gather
int calculate_col_hops(int m) {
    int total_hops = 0;
    for (int k = 0; (1 << k) < m; ++k) {
        total_hops += (1 << k);
    }
    return total_hops;
}

// Function to calculate total data moved for row-wise gather or reduce
int calculate_row_data_moved(int n, int data_size) {
    int total_data = 0;
    for (int k = 0; (1 << k) < n; ++k) {
        total_data += (1 << k) * data_size;
    }
    return total_data;
}

// Function to calculate total data moved for column-wise gather or reduce
int calculate_col_data_moved(int m, int data_size, int n) {
    int total_data = 0;
    for (int k = 0; (1 << k) < m; ++k) {
        total_data += (1 << k) * data_size * n;
    }
    return total_data;
}

// Function to calculate total hops for row-wise broadcast
int calculate_row_broadcast_hops(int n) {
    return calculate_row_hops(n);
}

// Function to calculate total hops for column-wise broadcast
int calculate_col_broadcast_hops(int m) {
    return calculate_col_hops(m);
}

// Function to calculate total data moved for row-wise broadcast
int calculate_row_broadcast_data_moved(int n, int data_size) {
    return calculate_row_data_moved(n, data_size);
}

// Function to calculate total data moved for column-wise broadcast
int calculate_col_broadcast_data_moved(int m, int data_size, int n) {
    return calculate_col_data_moved(m, data_size, n);
}

int main() {
    int m, n, data_size;
    cout << "Enter number of rows (m): ";
    cin >> m;
    cout << "Enter number of columns (n): ";
    cin >> n;
    cout << "Enter data size per node: ";
    cin >> data_size;

    // Gather Phase (for All-Gather operation)
    int row_hops = m * calculate_row_hops(n);
    int col_hops = calculate_col_hops(m);
    int row_data_moved = m * calculate_row_data_moved(n, data_size);
    int col_data_moved = calculate_col_data_moved(m, data_size, n);

    int total_gather_hops = row_hops + col_hops;
    int total_gather_data_moved = row_data_moved + col_data_moved;

    // Broadcast Phase (for All-Gather operation)
    int broadcast_row_hops = m * calculate_row_broadcast_hops(n);
    int broadcast_col_hops = calculate_col_broadcast_hops(m);
    int broadcast_row_data_moved = m * calculate_row_broadcast_data_moved(n, data_size);
    int broadcast_col_data_moved = calculate_col_broadcast_data_moved(m, data_size, n);

    int total_broadcast_hops = broadcast_row_hops + broadcast_col_hops;
    int total_broadcast_data_moved = broadcast_row_data_moved + broadcast_col_data_moved;

    int total_all_gather_hops = total_gather_hops + total_broadcast_hops;
    int total_all_gather_data_moved = total_gather_data_moved + total_broadcast_data_moved;

    // Reduce Phase
    int total_reduce_hops = total_gather_hops;
    int total_reduce_data_moved = total_gather_data_moved;

    cout << "\nTotal number of hops for Gather Operation: " << total_gather_hops << endl;
    cout << "Total data moved for Gather Operation: " << total_gather_data_moved << " units" << endl;
    cout << "\nTotal number of hops for Broadcast Operation: " << total_broadcast_hops << endl;
    cout << "Total data moved for Broadcast Operation: " << total_broadcast_data_moved << " units" << endl;
    cout << "\nTotal number of hops for All-Gather Operation: " << total_all_gather_hops << endl;
    cout << "Total data moved for All-Gather Operation: " << total_all_gather_data_moved << " units" << endl;
    cout << "\nTotal number of hops for Reduce Operation: " << total_reduce_hops << endl;
    cout << "Total data moved for Reduce Operation: " << total_reduce_data_moved << " units" << endl;

    return 0;
}



// Function to calculate hops and data moved for recursive doubling broadcast
void recursiveDoublingMetrics(int M, int N, int message_size, int &total_hops, int &total_data_moved) {
    total_hops = 0;

    // Horizontal broadcast phase (within the row)
    int horiz_steps = ceil(log2(N));
    int horiz_hops = 0;
    for (int step = 0; step < horiz_steps; ++step) {
        int num_senders = min(N, (1 << step));
        horiz_hops += num_senders; // each sender sends data to exactly one neighbor (1 hop)
    }
    // total horizontal hops across one row
    total_hops += horiz_hops;

    // Vertical broadcast phase (column-wise)
    int vert_steps = ceil(log2(M));
    int vert_hops = 0;
    for (int step = 0; step < vert_steps; ++step) {
        int rows_with_data = min(M, (1 << step));
        vert_hops += rows_with_data * N; // each processor with data sends to exactly one processor below
    }
    total_hops += vert_hops;

    // Total data moved is total hops times message size
    total_data_moved = total_hops * message_size;
}

#include <iostream>
#include <cmath>

using namespace std;

// Function to calculate hops and data moved for recursive doubling broadcast
void recursiveDoublingMetrics(int M, int N, int message_size, int &total_hops, int &total_data_moved) {
    total_hops = 0;

    // Horizontal broadcast phase (within the row)
    int horiz_steps = ceil(log2(N));
    int horiz_hops = 0;
    for (int step = 0; step < horiz_steps; ++step) {
        int num_senders = min(N, (1 << step));
        horiz_hops += num_senders; // each sender sends data to exactly one neighbor (1 hop)
    }
    // total horizontal hops across one row
    total_hops += horiz_hops;

    // Vertical broadcast phase (column-wise)
    int vert_steps = ceil(log2(M));
    int vert_hops = 0;
    for (int step = 0; step < vert_steps; ++step) {
        int rows_with_data = min(M, (1 << step));
        vert_hops += rows_with_data * N; // each processor with data sends to exactly one processor below
    }
    total_hops += vert_hops;

    // Total data moved is total hops times message size
    total_data_moved = total_hops * message_size;
}

int main() {
    int M, N, message_size;

    // Example Input:
    cout << "Enter M (rows): ";
    cin >> M;
    cout << "Enter N (columns): ";
    cin >> N;
    cout << "Enter message size (in bytes): ";
    cin >> message_size;

    int total_hops, total_data_moved;

    recursiveDoublingMetrics(M, N, message_size, total_hops, total_data_moved);

    cout << "\n--- Recursive Doubling Broadcast Metrics ---\n";
    cout << "Total number of hops: " << total_hops << endl;
    cout << "Total data moved (bytes): " << total_data_moved << endl;

    return 0;
}