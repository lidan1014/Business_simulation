import pandas as pd
import business_simulation_project

simulation_data = business_simulation_project.Business_simulation(p=0.5, num_customer=1000, session_range=100,
                                      a=2, b=5, p_0=200, p_1=400, p_2=8000,
                                      num_product=1000, beta=0.1, x0=8)

# part1:
customer_information_df = simulation_data.simulate_customer_information()
print(customer_information_df[0:5])

# part2:
product_information_df = simulation_data.simulate_product_information()
print(product_information_df[0:5])

# part3:
shopping_df = simulation_data.simulate_shopping_information(customer_information_df)
print(shopping_df[0:5])

# part4,5:
total_0_1_df = simulation_data.simulate_total_buying01(shopping_df, product_information_df)
simulated_buying_0_1 = pd.concat([customer_information_df, total_0_1_df], axis=1)
print(simulated_buying_0_1[0:5])

# part6:
whole_data = simulation_data.simulate_whole_data(simulated_buying_0_1, shopping_df)
print(whole_data.sort_values(by=['Session'], ascending=False))
print(pd.value_counts(shopping_df['customer_id'])[0:5])
