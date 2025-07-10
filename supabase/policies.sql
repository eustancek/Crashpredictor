-- Crash Values Policies
CREATE POLICY "Users can insert crash values" ON crash_values
FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Users can view own crash values" ON crash_values
FOR SELECT USING (auth.uid() = user_id);

-- Multipliers Policies
CREATE POLICY "Users can insert multipliers" ON multipliers
FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Users can view own multipliers" ON multipliers
FOR SELECT USING (auth.uid() = user_id);

-- Model Versions (Public Read)
CREATE POLICY "All users can read model versions" ON model_versions
FOR SELECT USING (true);

-- Training History Policies
CREATE POLICY "Users can insert training history" ON training_history
FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Users can view own training history" ON training_history
FOR SELECT USING (auth.uid() = user_id);

-- Accuracy Tracking Policies
CREATE POLICY "Users can insert accuracy tracking" ON accuracy_tracking
FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY "Users can view own accuracy tracking" ON accuracy_tracking
FOR SELECT USING (auth.uid() = user_id);